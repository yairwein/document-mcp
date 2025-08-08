"""Test configuration and fixtures."""

import asyncio
import tempfile
import pytest
import pytest_asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config
from src.parser import DocumentParser
from src.llm import LocalLLM, DocumentProcessor
from src.indexer import DocumentIndexer
from src.tools import DocumentTools
from src.main import DocumentIndexerService


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration."""
    return Config(
        watch_folders=[],
        lancedb_path=temp_dir / "test_index",
        llm_model="llama3.2:3b",
        chunk_size=500,
        chunk_overlap=100,
        embedding_model="all-MiniLM-L6-v2",
        file_extensions=[".pdf", ".docx", ".doc", ".txt", ".md"],
        max_file_size_mb=100,
        ollama_base_url="http://localhost:11434"
    )


@pytest.fixture
def document_parser():
    """Create a document parser."""
    return DocumentParser(chunk_size=500, chunk_overlap=100)


@pytest_asyncio.fixture
async def llm(test_config):
    """Create and initialize an LLM."""
    llm = LocalLLM(model=test_config.llm_model, base_url=test_config.ollama_base_url)
    await llm.initialize()
    yield llm
    await llm.close()


@pytest_asyncio.fixture
async def document_processor(llm):
    """Create a document processor."""
    return DocumentProcessor(llm)


@pytest_asyncio.fixture
async def document_indexer(test_config):
    """Create and initialize a document indexer."""
    test_config.ensure_dirs()
    indexer = DocumentIndexer(
        db_path=test_config.lancedb_path,
        embedding_model=test_config.embedding_model
    )
    await indexer.initialize()
    yield indexer
    await indexer.close()


@pytest_asyncio.fixture
async def document_tools(document_indexer, document_parser, document_processor):
    """Create document tools."""
    return DocumentTools(document_indexer, document_parser, document_processor)


@pytest_asyncio.fixture
async def service(test_config):
    """Create a document indexer service."""
    service = DocumentIndexerService()
    service.config = test_config
    await service.initialize()
    yield service
    await service.stop()


@pytest.fixture
def sample_text_content():
    """Sample text content for testing."""
    return """
    This is a test document for the MCP Document Indexer.
    
    The system can index various document types including PDF, Word, and text files.
    It uses LanceDB for vector storage and semantic search capabilities.
    
    Features include:
    - Real-time document monitoring
    - Automatic indexing
    - Semantic search
    - Document summarization
    
    This is a longer paragraph to test the chunking functionality. The system should
    be able to split this text into multiple chunks with some overlap to maintain
    context between chunks. This helps with better retrieval during search operations.
    
    Legal Terms and Conditions:
    This document contains confidential information and is subject to non-disclosure
    agreements. Any unauthorized use or distribution is strictly prohibited.
    """


@pytest.fixture
def sample_legal_content():
    """Sample legal document content."""
    return """
    MUTUAL NON-DISCLOSURE AGREEMENT
    
    This Mutual Confidential Disclosure Agreement (the "Agreement"), effective as of 
    January 1, 2024, governs the disclosure of information by and between 
    Acme Corporation ("Acme") and Beta Industries Inc. ("Beta").
    
    1. CONFIDENTIAL INFORMATION
    For purposes of this Agreement, "Confidential Information" means any and all
    non-public, confidential or proprietary information.
    
    2. OBLIGATIONS
    Each party agrees to maintain the confidentiality of all Confidential Information
    received from the other party.
    
    3. TERM
    This Agreement shall remain in effect for a period of five (5) years.
    """


@pytest.fixture
def sample_text_file(temp_dir, sample_text_content):
    """Create a sample text file."""
    file_path = temp_dir / "test_document.txt"
    file_path.write_text(sample_text_content)
    return file_path


@pytest.fixture
def sample_legal_file(temp_dir, sample_legal_content):
    """Create a sample legal document file."""
    file_path = temp_dir / "nda_agreement.txt"
    file_path.write_text(sample_legal_content)
    return file_path


@pytest.fixture
def multiple_test_files(temp_dir, sample_text_content, sample_legal_content):
    """Create multiple test files."""
    files = []
    
    # Text file
    text_file = temp_dir / "document1.txt"
    text_file.write_text(sample_text_content)
    files.append(text_file)
    
    # Legal file
    legal_file = temp_dir / "contract.txt"
    legal_file.write_text(sample_legal_content)
    files.append(legal_file)
    
    # Technical file
    tech_content = """
    API Documentation
    
    This document describes the REST API endpoints for our service.
    
    GET /api/users - Retrieve all users
    POST /api/users - Create a new user
    PUT /api/users/{id} - Update a user
    DELETE /api/users/{id} - Delete a user
    
    Authentication is required for all endpoints using JWT tokens.
    """
    tech_file = temp_dir / "api_docs.txt"
    tech_file.write_text(tech_content)
    files.append(tech_file)
    
    return files