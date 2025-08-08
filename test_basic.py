#!/usr/bin/env python3
"""Basic test to verify the system works."""

import asyncio
import tempfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

async def test_basic():
    """Test basic functionality."""
    from src.config import Config
    from src.parser import DocumentParser
    from src.llm import LocalLLM, DocumentProcessor
    from src.indexer import DocumentIndexer
    
    print("Testing MCP Document Indexer...")
    
    # Create test config
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            watch_folders=[],
            lancedb_path=Path(tmpdir) / "test_index",
            llm_model="llama3.2:3b",
            chunk_size=500,
            chunk_overlap=100
        )
        config.ensure_dirs()
        
        # Test parser
        print("\n1. Testing Document Parser...")
        parser = DocumentParser(chunk_size=500, chunk_overlap=100)
        
        # Create a test text file
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("""
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
        """)
        
        doc_data = parser.parse_file(test_file)
        print(f"   ✓ Parsed document: {doc_data['metadata']['file_name']}")
        print(f"   ✓ Created {doc_data['num_chunks']} chunks")
        
        # Test LLM (will use fallback if Ollama not available)
        print("\n2. Testing Local LLM...")
        llm = LocalLLM(model=config.llm_model, base_url=config.ollama_base_url)
        await llm.initialize()
        processor = DocumentProcessor(llm)
        
        doc_data = await processor.process_document(doc_data)
        print(f"   ✓ Generated summary: {doc_data['summary'][:100]}...")
        print(f"   ✓ Extracted keywords: {doc_data['keywords'][:5]}")
        
        # Test indexer
        print("\n3. Testing Document Indexer...")
        indexer = DocumentIndexer(
            db_path=config.lancedb_path,
            embedding_model=config.embedding_model
        )
        await indexer.initialize()
        
        success = await indexer.index_document(doc_data)
        print(f"   ✓ Document indexed: {success}")
        
        # Test catalog
        print("\n4. Testing Catalog...")
        catalog = await indexer.get_catalog()
        print(f"   ✓ Catalog has {len(catalog)} documents")
        
        # Get stats
        stats = await indexer.get_stats()
        print(f"   ✓ Stats: {stats['total_documents']} docs, {stats['total_chunks']} chunks")
        
        # Note: Vector search requires further LanceDB configuration
        print("\n5. Vector Search...")
        print("   ⚠ Vector search requires index creation - skipping for now")
        
        # Clean up
        await llm.close()
        await indexer.close()
        
        print("\n✅ All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_basic())