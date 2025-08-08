"""Unit tests for document indexer."""

import pytest
import asyncio
from pathlib import Path
from src.indexer import DocumentIndexer


class TestDocumentIndexer:
    """Test document indexer functionality."""

    @pytest.mark.unit
    def test_init(self, temp_dir):
        """Test indexer initialization."""
        indexer = DocumentIndexer(temp_dir / "test_db", "all-MiniLM-L6-v2")
        assert indexer.db_path == temp_dir / "test_db"
        assert indexer.embedding_model_name == "all-MiniLM-L6-v2"
        assert indexer.embedding_model is None
        assert indexer.db is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialization(self, document_indexer):
        """Test indexer initialization with database setup."""
        # Should be initialized by fixture
        assert document_indexer.embedding_model is not None
        assert document_indexer.db is not None
        assert document_indexer.catalog_table is not None
        assert document_indexer.chunks_table is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_index_document(self, document_indexer, document_parser, document_processor, sample_text_file):
        """Test document indexing."""
        # Parse and process document
        doc_data = document_parser.parse_file(sample_text_file)
        doc_data = await document_processor.process_document(doc_data)
        
        # Index document
        result = await document_indexer.index_document(doc_data)
        assert result is True  # Should succeed for new document

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_index_same_document_twice(self, document_indexer, document_parser, document_processor, sample_text_file):
        """Test indexing the same document twice (should detect no changes)."""
        # Parse and process document
        doc_data = document_parser.parse_file(sample_text_file)
        doc_data = await document_processor.process_document(doc_data)
        
        # Index document first time
        result1 = await document_indexer.index_document(doc_data)
        assert result1 is True
        
        # Index same document again
        result2 = await document_indexer.index_document(doc_data)
        assert result2 is False  # Should detect no changes

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_catalog(self, document_indexer, document_parser, document_processor, sample_text_file):
        """Test getting document catalog."""
        # Index a document first
        doc_data = document_parser.parse_file(sample_text_file)
        doc_data = await document_processor.process_document(doc_data)
        await document_indexer.index_document(doc_data)
        
        # Get catalog
        catalog = await document_indexer.get_catalog()
        
        assert len(catalog) == 1
        doc = catalog[0]
        assert doc['file_name'] == sample_text_file.name
        assert doc['file_path'] == str(sample_text_file)
        assert doc['file_type'] == '.txt'
        assert 'summary' in doc
        assert 'keywords' in doc

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_catalog_with_pagination(self, document_indexer, document_parser, document_processor, multiple_test_files):
        """Test catalog pagination."""
        # Index multiple documents
        for file in multiple_test_files:
            doc_data = document_parser.parse_file(file)
            doc_data = await document_processor.process_document(doc_data)
            await document_indexer.index_document(doc_data)
        
        # Test pagination
        page1 = await document_indexer.get_catalog(skip=0, limit=2)
        page2 = await document_indexer.get_catalog(skip=2, limit=2)
        
        assert len(page1) == 2
        assert len(page2) == 1  # Only 3 files total
        
        # Should not have overlapping documents
        page1_paths = {doc['file_path'] for doc in page1}
        page2_paths = {doc['file_path'] for doc in page2}
        assert len(page1_paths.intersection(page2_paths)) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_document_info(self, document_indexer, document_parser, document_processor, sample_text_file):
        """Test getting specific document info."""
        # Index a document
        doc_data = document_parser.parse_file(sample_text_file)
        doc_data = await document_processor.process_document(doc_data)
        await document_indexer.index_document(doc_data)
        
        # Get document info
        info = await document_indexer.get_document_info(str(sample_text_file))
        
        assert info is not None
        assert info['file_path'] == str(sample_text_file)
        assert info['file_name'] == sample_text_file.name
        assert 'file_hash' in info
        assert 'file_size' in info
        assert 'summary' in info
        assert 'keywords' in info
        assert 'total_chunks' in info

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_document_info_nonexistent(self, document_indexer):
        """Test getting info for nonexistent document."""
        info = await document_indexer.get_document_info("/nonexistent/file.txt")
        assert info is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_remove_document(self, document_indexer, document_parser, document_processor, sample_text_file):
        """Test document removal."""
        # Index a document
        doc_data = document_parser.parse_file(sample_text_file)
        doc_data = await document_processor.process_document(doc_data)
        await document_indexer.index_document(doc_data)
        
        # Verify it exists
        catalog = await document_indexer.get_catalog()
        assert len(catalog) == 1
        
        # Remove document
        removed = await document_indexer.remove_document(str(sample_text_file))
        assert removed is True
        
        # Verify it's gone
        catalog = await document_indexer.get_catalog()
        assert len(catalog) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_remove_nonexistent_document(self, document_indexer):
        """Test removing nonexistent document."""
        removed = await document_indexer.remove_document("/nonexistent/file.txt")
        assert removed is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_stats(self, document_indexer, document_parser, document_processor, multiple_test_files):
        """Test getting indexer statistics."""
        # Index multiple documents
        for file in multiple_test_files:
            doc_data = document_parser.parse_file(file)
            doc_data = await document_processor.process_document(doc_data)
            await document_indexer.index_document(doc_data)
        
        stats = await document_indexer.get_stats()
        
        assert 'total_documents' in stats
        assert 'total_chunks' in stats
        assert 'total_size_bytes' in stats
        assert 'total_chars' in stats
        assert 'total_tokens' in stats
        assert 'file_types' in stats
        assert 'db_path' in stats
        
        assert stats['total_documents'] == len(multiple_test_files)
        assert stats['total_chunks'] > 0
        assert stats['total_size_bytes'] > 0
        assert '.txt' in stats['file_types']

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_stats(self, document_indexer):
        """Test stats for empty database."""
        stats = await document_indexer.get_stats()
        
        assert stats['total_documents'] == 0
        assert stats['total_chunks'] == 0
        assert stats['total_size_bytes'] == 0
        assert stats['total_chars'] == 0
        assert stats['total_tokens'] == 0
        assert stats['file_types'] == {}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_document_update(self, document_indexer, document_parser, document_processor, temp_dir):
        """Test updating a document (same path, different content)."""
        test_file = temp_dir / "update_test.txt"
        
        # Create initial document
        test_file.write_text("Original content")
        doc_data1 = document_parser.parse_file(test_file)
        doc_data1 = await document_processor.process_document(doc_data1)
        result1 = await document_indexer.index_document(doc_data1)
        assert result1 is True
        
        # Update document content
        test_file.write_text("Updated content with more information")
        doc_data2 = document_parser.parse_file(test_file)
        doc_data2 = await document_processor.process_document(doc_data2)
        result2 = await document_indexer.index_document(doc_data2)
        assert result2 is True  # Should detect changes and update
        
        # Verify updated content
        info = await document_indexer.get_document_info(str(test_file))
        assert info['file_hash'] == doc_data2['metadata']['file_hash']

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_indexing(self, document_indexer, document_parser, document_processor, multiple_test_files):
        """Test concurrent document indexing."""
        # Prepare document data
        doc_data_list = []
        for file in multiple_test_files:
            doc_data = document_parser.parse_file(file)
            doc_data = await document_processor.process_document(doc_data)
            doc_data_list.append(doc_data)
        
        # Index concurrently
        tasks = [document_indexer.index_document(doc_data) for doc_data in doc_data_list]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(result is True for result in results)
        
        # Verify all documents are indexed
        catalog = await document_indexer.get_catalog()
        assert len(catalog) == len(multiple_test_files)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_embedding_generation(self, document_indexer):
        """Test embedding generation."""
        text = "This is a test text for embedding generation"
        
        # This is a protected method but we can test it exists and works
        embedding = await document_indexer._generate_embedding(text)
        
        import numpy as np
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
        assert embedding.dtype == np.float32

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self, document_indexer):
        """Test batch embedding generation."""
        texts = [
            "First test text",
            "Second test text",
            "Third test text"
        ]
        
        embeddings = await document_indexer._generate_embeddings_batch(texts)
        
        import numpy as np
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) > 0
            assert embedding.dtype == np.float32

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_database_persistence(self, test_config, document_parser, document_processor, sample_text_file):
        """Test that data persists across indexer instances."""
        # Create first indexer instance and index document
        indexer1 = DocumentIndexer(test_config.lancedb_path, test_config.embedding_model)
        await indexer1.initialize()
        
        doc_data = document_parser.parse_file(sample_text_file)
        doc_data = await document_processor.process_document(doc_data)
        await indexer1.index_document(doc_data)
        
        catalog1 = await indexer1.get_catalog()
        assert len(catalog1) == 1
        
        await indexer1.close()
        
        # Create second indexer instance and check data persists
        indexer2 = DocumentIndexer(test_config.lancedb_path, test_config.embedding_model)
        await indexer2.initialize()
        
        catalog2 = await indexer2.get_catalog()
        assert len(catalog2) == 1
        assert catalog2[0]['file_path'] == catalog1[0]['file_path']
        
        await indexer2.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close(self, document_indexer):
        """Test indexer cleanup."""
        await document_indexer.close()
        # Should not raise any exceptions

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_handling_invalid_document(self, document_indexer):
        """Test error handling for invalid document data."""
        invalid_doc_data = {
            'metadata': {'file_path': '/invalid/path'},
            # Missing required fields
        }
        
        with pytest.raises((KeyError, AttributeError, ValueError)):
            await document_indexer.index_document(invalid_doc_data)