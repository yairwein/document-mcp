"""File monitoring system for automatic document indexing."""

import asyncio
import logging
from pathlib import Path
from typing import Set, List, Callable, Optional
from datetime import datetime, timedelta
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
import queue

logger = logging.getLogger(__name__)


class DocumentEventHandler(FileSystemEventHandler):
    """Handle file system events for document monitoring."""
    
    def __init__(self, file_extensions: List[str], event_queue: queue.Queue):
        self.file_extensions = set(file_extensions)
        self.event_queue = event_queue
        self.processed_events = {}  # Track processed events to avoid duplicates
        self.debounce_seconds = 2  # Wait time before processing
    
    def should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed."""
        path = Path(file_path)
        
        # Skip hidden files and directories
        if any(part.startswith('.') for part in path.parts):
            return False
        
        # Skip temporary files
        if path.name.startswith('~') or path.name.startswith('._'):
            return False
        
        # Check extension
        return path.suffix.lower() in self.file_extensions
    
    def debounce_event(self, event_key: str) -> bool:
        """Check if event should be processed (debouncing)."""
        now = datetime.now()
        
        if event_key in self.processed_events:
            last_time = self.processed_events[event_key]
            if (now - last_time).total_seconds() < self.debounce_seconds:
                return False
        
        self.processed_events[event_key] = now
        
        # Clean old entries
        cutoff = now - timedelta(minutes=5)
        self.processed_events = {
            k: v for k, v in self.processed_events.items()
            if v > cutoff
        }
        
        return True
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation."""
        if not event.is_directory and self.should_process_file(event.src_path):
            event_key = f"create:{event.src_path}"
            if self.debounce_event(event_key):
                logger.info(f"New file detected: {event.src_path}")
                self.event_queue.put(('create', event.src_path))
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification."""
        if not event.is_directory and self.should_process_file(event.src_path):
            event_key = f"modify:{event.src_path}"
            if self.debounce_event(event_key):
                logger.info(f"File modified: {event.src_path}")
                self.event_queue.put(('modify', event.src_path))
    
    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion."""
        if not event.is_directory:
            # We don't check extension for deletion as file might be gone
            event_key = f"delete:{event.src_path}"
            if self.debounce_event(event_key):
                logger.info(f"File deleted: {event.src_path}")
                self.event_queue.put(('delete', event.src_path))
    
    def on_moved(self, event: FileSystemEvent):
        """Handle file move/rename."""
        if not event.is_directory:
            # Treat as delete + create
            if self.should_process_file(event.dest_path):
                logger.info(f"File moved: {event.src_path} -> {event.dest_path}")
                self.event_queue.put(('delete', event.src_path))
                self.event_queue.put(('create', event.dest_path))


class FileMonitor:
    """Monitor directories for document changes."""
    
    def __init__(self, 
                 watch_folders: List[Path],
                 file_extensions: List[str],
                 callback: Optional[Callable] = None):
        self.watch_folders = watch_folders
        self.file_extensions = file_extensions
        self.callback = callback
        self.observer = None
        self.event_queue = queue.Queue()
        self.running = False
        self._process_task = None
    
    def start(self):
        """Start monitoring."""
        if self.running:
            logger.warning("Monitor already running")
            return
        
        if not self.watch_folders:
            logger.warning("No folders to watch")
            return
        
        self.observer = Observer()
        handler = DocumentEventHandler(self.file_extensions, self.event_queue)
        
        for folder in self.watch_folders:
            if folder.exists() and folder.is_dir():
                logger.info(f"Watching folder: {folder}")
                self.observer.schedule(handler, str(folder), recursive=True)
            else:
                logger.warning(f"Folder not found or not a directory: {folder}")
        
        self.observer.start()
        self.running = True
        logger.info("File monitoring started")
    
    def stop(self):
        """Stop monitoring."""
        if not self.running:
            return
        
        self.running = False
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)
            self.observer = None
        
        logger.info("File monitoring stopped")
    
    async def process_events(self):
        """Process file events asynchronously."""
        while self.running:
            try:
                # Check for events with timeout
                try:
                    event = self.event_queue.get(timeout=1)
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
                
                event_type, file_path = event
                
                if self.callback:
                    try:
                        await self.callback(event_type, file_path)
                    except Exception as e:
                        logger.error(f"Error processing event {event_type} for {file_path}: {e}")
                
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(1)
    
    async def scan_existing_files(self) -> List[Path]:
        """Scan for existing files in watched folders."""
        all_files = []
        
        for folder in self.watch_folders:
            if not folder.exists() or not folder.is_dir():
                continue
            
            # Find all matching files
            for ext in self.file_extensions:
                pattern = f"**/*{ext}"
                files = list(folder.glob(pattern))
                
                # Filter out hidden and temp files
                files = [
                    f for f in files
                    if not any(part.startswith('.') for part in f.parts)
                    and not f.name.startswith('~')
                    and not f.name.startswith('._')
                ]
                
                all_files.extend(files)
        
        # Remove duplicates and sort
        all_files = sorted(set(all_files))
        logger.info(f"Found {len(all_files)} existing files to index")
        
        return all_files
    
    def get_queue_size(self) -> int:
        """Get number of pending events."""
        return self.event_queue.qsize()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class IndexingQueue:
    """Manage document indexing queue with priorities."""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.queue = asyncio.PriorityQueue()
        self.processing = set()
        self.processed = set()
        self.failed = {}
        
    async def add_file(self, file_path: Path, priority: int = 5):
        """Add file to indexing queue."""
        # Priority: 1=highest, 10=lowest
        # New files get higher priority than existing
        await self.queue.put((priority, str(file_path)))
    
    async def get_next(self) -> Optional[Path]:
        """Get next file to process."""
        try:
            priority, file_path = await asyncio.wait_for(
                self.queue.get(), timeout=1.0
            )
            
            # Skip if already processing or recently processed
            if file_path in self.processing or file_path in self.processed:
                return None
            
            self.processing.add(file_path)
            return Path(file_path)
            
        except asyncio.TimeoutError:
            return None
    
    def mark_complete(self, file_path: Path):
        """Mark file as processed."""
        file_str = str(file_path)
        self.processing.discard(file_str)
        self.processed.add(file_str)
        
        # Clean old entries if too many
        if len(self.processed) > 1000:
            self.processed = set(list(self.processed)[-500:])
    
    def mark_failed(self, file_path: Path, error: str):
        """Mark file as failed."""
        file_str = str(file_path)
        self.processing.discard(file_str)
        self.failed[file_str] = {
            'error': error,
            'time': datetime.now().isoformat()
        }
    
    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            'queued': self.queue.qsize(),
            'processing': len(self.processing),
            'processed': len(self.processed),
            'failed': len(self.failed)
        }