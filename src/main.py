#!/usr/bin/env python3
"""MCP Document Indexer - Main entry point."""

import asyncio
import logging
import signal
import sys
import subprocess
import time
import os
from pathlib import Path
from typing import Optional
from fastmcp import FastMCP

from .config import get_config
from .parser import DocumentParser, is_supported_file
from .llm import LocalLLM, DocumentProcessor
from .indexer import DocumentIndexer
from .monitor import FileMonitor, IndexingQueue
from .tools import DocumentTools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentIndexerService:
    """Main service orchestrating document indexing."""
    
    def __init__(self):
        self.config = get_config()
        self.parser = None
        self.llm = None
        self.processor = None
        self.indexer = None
        self.monitor = None
        self.indexing_queue = None
        self.tools = None
        self.mcp = None
        self.running = False
        self._indexing_task = None
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Document Indexer Service...")
        
        # Initialize components
        self.parser = DocumentParser(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        self.llm = LocalLLM(
            model=self.config.llm_model,
            base_url=self.config.ollama_base_url
        )
        await self.llm.initialize()
        
        self.processor = DocumentProcessor(self.llm)
        
        self.indexer = DocumentIndexer(
            db_path=self.config.lancedb_path,
            embedding_model=self.config.embedding_model
        )
        await self.indexer.initialize()
        
        self.indexing_queue = IndexingQueue(max_concurrent=5)
        
        # Initialize MCP tools
        self.tools = DocumentTools(self.indexer, self.parser, self.processor)
        
        # Initialize file monitor if folders configured
        if self.config.watch_folders:
            self.monitor = FileMonitor(
                watch_folders=self.config.watch_folders,
                file_extensions=self.config.file_extensions,
                callback=self.handle_file_event
            )
        
        logger.info("Service initialized successfully")
    
    async def handle_file_event(self, event_type: str, file_path: str):
        """Handle file system events."""
        file_path = Path(file_path)
        
        if event_type == 'delete':
            # Remove from index
            await self.indexer.remove_document(str(file_path))
        elif event_type in ['create', 'modify']:
            # Add to indexing queue
            priority = 3 if event_type == 'create' else 5
            await self.indexing_queue.add_file(file_path, priority)
    
    async def index_file(self, file_path: Path) -> bool:
        """Index a single file."""
        try:
            logger.info(f"  → Starting processing: {file_path.name}")
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_file_size_mb:
                logger.warning(f"  → Skipping: File too large ({file_size_mb:.1f}MB > {self.config.max_file_size_mb}MB)")
                return False
            
            logger.info(f"  → [1/4] Parsing document ({file_size_mb:.2f}MB)...")
            
            # Parse document
            doc_data = self.parser.parse_file(file_path)
            logger.info(f"  → [1/4] ✓ Parsed: {doc_data['total_chars']:,} chars, {doc_data['num_chunks']} chunks, {doc_data['total_tokens']:,} tokens")
            
            # Process with LLM
            logger.info(f"  → [2/4] Processing with LLM ({self.config.llm_model})...")
            doc_data = await self.processor.process_document(doc_data)
            logger.info(f"  → [2/4] ✓ Generated summary ({len(doc_data.get('summary', ''))} chars) and {len(doc_data.get('keywords', []))} keywords")
            
            # Index document
            logger.info(f"  → [3/4] Generating embeddings and indexing to LanceDB...")
            success = await self.indexer.index_document(doc_data)
            
            if success:
                logger.info(f"  → [4/4] ✓ Successfully indexed document with {doc_data['num_chunks']} chunks")
            else:
                logger.info(f"  → [4/4] ⟳ Document unchanged (already indexed)")
            
            return success
            
        except Exception as e:
            logger.error(f"Error indexing {file_path}: {e}")
            self.indexing_queue.mark_failed(file_path, str(e))
            return False
    
    async def process_indexing_queue(self):
        """Process files in the indexing queue."""
        processed_count = 0
        total_to_process = 0
        
        while self.running:
            try:
                # Get next file from queue
                file_path = await self.indexing_queue.get_next()
                
                if file_path:
                    # Log progress
                    queue_stats = self.indexing_queue.get_stats()
                    if processed_count == 0:
                        total_to_process = queue_stats['queued'] + queue_stats['processing']
                    
                    processed_count += 1
                    logger.info(f"[{processed_count}/{total_to_process}] Processing: {file_path.name}")
                    
                    success = await self.index_file(file_path)
                    if success:
                        self.indexing_queue.mark_complete(file_path)
                        logger.info(f"[{processed_count}/{total_to_process}] ✓ Indexed: {file_path.name}")
                    else:
                        self.indexing_queue.mark_complete(file_path)
                        logger.info(f"[{processed_count}/{total_to_process}] ⟳ Unchanged: {file_path.name}")
                    
                    # Log queue status every 10 files
                    if processed_count % 10 == 0:
                        logger.info(f"Progress: {processed_count} processed, {queue_stats['queued']} remaining in queue")
                else:
                    # No files to process, wait a bit
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error processing file: {e}")
                if file_path:
                    self.indexing_queue.mark_failed(file_path, str(e))
                await asyncio.sleep(1)
    
    async def initial_scan(self):
        """Perform initial scan of watched folders."""
        if not self.monitor:
            return
        
        logger.info("Starting initial document scan...")
        existing_files = await self.monitor.scan_existing_files()
        
        # Limit initial scan for large folders
        max_initial_files = int(os.getenv("MAX_INITIAL_FILES", "50"))
        if len(existing_files) > max_initial_files:
            logger.warning(f"Found {len(existing_files)} files, limiting initial scan to {max_initial_files} most recent files")
            logger.info("Set MAX_INITIAL_FILES env var to increase this limit")
            # Sort by modification time and take most recent
            existing_files = sorted(existing_files, key=lambda x: x.stat().st_mtime, reverse=True)[:max_initial_files]
        
        # Add files to indexing queue with lower priority
        added_count = 0
        for file_path in existing_files:
            if is_supported_file(file_path, self.config.file_extensions):
                await self.indexing_queue.add_file(file_path, priority=8)
                added_count += 1
        
        logger.info(f"Added {added_count} files to indexing queue")
    
    async def start(self):
        """Start the service."""
        if self.running:
            return
        
        self.running = True
        
        # Start file monitor
        if self.monitor:
            self.monitor.start()
            
            # Start monitor event processing
            asyncio.create_task(self.monitor.process_events())
            
            # Perform initial scan
            await self.initial_scan()
        
        # Start indexing queue processor
        self._indexing_task = asyncio.create_task(self.process_indexing_queue())
        
        logger.info("Document Indexer Service started")
    
    async def stop(self):
        """Stop the service."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop monitor
        if self.monitor:
            self.monitor.stop()
        
        # Cancel indexing task
        if self._indexing_task:
            self._indexing_task.cancel()
            try:
                await self._indexing_task
            except asyncio.CancelledError:
                pass
        
        # Clean up resources
        if self.llm:
            await self.llm.close()
        if self.indexer:
            await self.indexer.close()
        
        logger.info("Document Indexer Service stopped")
    
    def setup_mcp_server(self) -> FastMCP:
        """Set up the MCP server with tools."""
        mcp = FastMCP("mcp-doc-indexer", instructions="Local document indexing and search with LanceDB")
        
        # Register tools
        mcp.tool(self.tools.search_documents)
        mcp.tool(self.tools.get_catalog)
        mcp.tool(self.tools.get_document_info)
        mcp.tool(self.tools.reindex_document)
        mcp.tool(self.tools.get_indexing_stats)
        
        return mcp


def ensure_ollama_running():
    """Ensure Ollama is running, start it if not."""
    try:
        # Check if Ollama is running
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True,
            timeout=2
        )
        if result.returncode == 0:
            logger.info("Ollama is already running")
            return True
    except Exception:
        pass
    
    # Try to start Ollama
    logger.info("Starting Ollama serve in background...")
    try:
        # Start ollama serve in background
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        # Wait a bit for it to start
        time.sleep(3)
        
        # Check if it's running now
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True,
            timeout=2
        )
        
        if result.returncode == 0:
            logger.info("Ollama serve started successfully")
            return True
        else:
            logger.warning("Ollama serve started but not responding yet")
            return False
            
    except FileNotFoundError:
        logger.error("Ollama not found. Please install from https://ollama.com/download")
        return False
    except Exception as e:
        logger.error(f"Failed to start Ollama: {e}")
        return False


async def main():
    """Main entry point."""
    # Ensure Ollama is running
    ensure_ollama_running()
    
    # Create service
    service = DocumentIndexerService()
    
    try:
        # Initialize service
        await service.initialize()
        
        # Start service
        await service.start()
        
        # Set up MCP server
        mcp = service.setup_mcp_server()
        
        # Setup signal handlers
        def signal_handler(sig, frame):
            logger.info("Shutting down...")
            asyncio.create_task(service.stop())
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info("MCP Document Indexer running...")
        logger.info(f"Watching folders: {service.config.watch_folders}")
        logger.info(f"Database path: {service.config.lancedb_path}")
        
        # Run the MCP server in stdio mode for Claude Desktop
        await mcp.run_stdio_async()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Service error: {e}", exc_info=True)
    finally:
        await service.stop()


def run():
    """Run the service."""
    asyncio.run(main())


if __name__ == "__main__":
    run()