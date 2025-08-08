"""Configuration management for MCP Document Indexer."""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class Config(BaseModel):
    """Configuration for the document indexer."""
    
    watch_folders: List[Path] = Field(
        default_factory=list,
        description="Folders to monitor for documents"
    )
    lancedb_path: Path = Field(
        default=Path("./vector_index"),
        description="Path to LanceDB storage"
    )
    llm_model: str = Field(
        default="llama3.2:3b",
        description="Ollama model for document summarization"
    )
    chunk_size: int = Field(
        default=1000,
        description="Size of text chunks in characters"
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between chunks in characters"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    file_extensions: List[str] = Field(
        default=[".pdf", ".docx", ".doc", ".txt", ".md", ".rtf"],
        description="File extensions to index"
    )
    max_file_size_mb: int = Field(
        default=100,
        description="Maximum file size in MB to process"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    batch_size: int = Field(
        default=10,
        description="Batch size for processing documents"
    )
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        watch_folders_str = os.getenv("WATCH_FOLDERS", "")
        watch_folders = []
        if watch_folders_str:
            for folder in watch_folders_str.split(","):
                folder = folder.strip()
                if folder:
                    path = Path(folder).expanduser().absolute()
                    if path.exists() and path.is_dir():
                        watch_folders.append(path)
                    else:
                        print(f"Warning: Folder {folder} does not exist or is not a directory")
        
        lancedb_path = os.getenv("LANCEDB_PATH", "./vector_index")
        llm_model = os.getenv("LLM_MODEL", "llama3.2:3b")
        chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        file_extensions_str = os.getenv("FILE_EXTENSIONS", ".pdf,.docx,.doc,.txt,.md,.rtf")
        file_extensions = [ext.strip() for ext in file_extensions_str.split(",")]
        
        max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        batch_size = int(os.getenv("BATCH_SIZE", "10"))
        
        return cls(
            watch_folders=watch_folders,
            lancedb_path=Path(lancedb_path).expanduser().absolute(),
            llm_model=llm_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            file_extensions=file_extensions,
            max_file_size_mb=max_file_size_mb,
            ollama_base_url=ollama_base_url,
            batch_size=batch_size
        )
    
    def ensure_dirs(self):
        """Ensure all required directories exist."""
        self.lancedb_path.mkdir(parents=True, exist_ok=True)


_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global configuration."""
    global _config
    if _config is None:
        _config = Config.from_env()
        _config.ensure_dirs()
    return _config