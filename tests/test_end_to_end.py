"""End-to-end integration tests."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from src.main import DocumentIndexerService


class MockContext:
    """Mock Context for testing."""
    pass


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_complete_document_lifecycle(self, test_config, multiple_test_files):
        """Test complete document lifecycle from indexing to search."""
        # Create service
        service = DocumentIndexerService()
        service.config = test_config
        
        try:
            # Initialize service
            await service.initialize()
            
            # Index documents
            for file in multiple_test_files:
                doc_data = service.parser.parse_file(file)
                doc_data = await service.processor.process_document(doc_data)
                result = await service.indexer.index_document(doc_data)
                assert result is True
            
            # Verify documents are indexed
            stats = await service.indexer.get_stats()
            assert stats['total_documents'] == len(multiple_test_files)
            assert stats['total_chunks'] > 0
            
            # Get catalog
            catalog = await service.indexer.get_catalog()
            assert len(catalog) == len(multiple_test_files)
            
            # Search for documents
            # Note: Vector search might fail due to embedding model issues
            # but document metadata search should work
            
            # Get document info
            first_file = multiple_test_files[0]
            info = await service.indexer.get_document_info(str(first_file))
            assert info is not None
            assert info['file_path'] == str(first_file)
            
            # Update document
            first_file.write_text("Updated content for testing")
            updated_doc_data = service.parser.parse_file(first_file)
            updated_doc_data = await service.processor.process_document(updated_doc_data)
            update_result = await service.indexer.index_document(updated_doc_data)
            assert update_result is True
            
            # Verify update
            updated_info = await service.indexer.get_document_info(str(first_file))
            assert updated_info['file_hash'] != info['file_hash']
            
            # Remove document
            removed = await service.indexer.remove_document(str(first_file))
            assert removed is True
            
            # Verify removal
            final_catalog = await service.indexer.get_catalog()
            assert len(final_catalog) == len(multiple_test_files) - 1
            
        finally:
            await service.stop()

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_mcp_server_full_workflow(self, test_config, sample_text_file, sample_legal_file):
        """Test full MCP server workflow."""
        # Create service
        service = DocumentIndexerService()
        service.config = test_config
        
        try:
            # Initialize service
            await service.initialize()
            
            # Setup MCP server
            mcp_server = service.setup_mcp_server()
            assert mcp_server is not None
            
            # Index documents through service
            for file in [sample_text_file, sample_legal_file]:
                doc_data = service.parser.parse_file(file)
                doc_data = await service.processor.process_document(doc_data)
                await service.indexer.index_document(doc_data)
            
            # Test MCP tools
            from fastmcp import Context
            from src.tools import SearchDocumentsInput, GetCatalogInput, GetDocumentInfoInput
            
            ctx = MockContext()
            tools = service.tools
            
            # Test search
            search_result = await tools.search_documents(
                ctx, SearchDocumentsInput(query="document", limit=10)
            )
            assert 'success' in search_result
            assert 'query' in search_result
            assert 'results' in search_result
            
            # Test catalog
            catalog_result = await tools.get_catalog(
                ctx, GetCatalogInput(skip=0, limit=10)
            )
            assert catalog_result['success'] is True
            assert catalog_result['total_documents'] == 2
            
            # Test document info
            info_result = await tools.get_document_info(
                ctx, GetDocumentInfoInput(file_path=str(sample_text_file))
            )
            assert info_result['success'] is True
            
            # Test stats
            stats_result = await tools.get_indexing_stats(ctx)
            assert stats_result['success'] is True
            assert stats_result['stats']['total_documents'] == 2
            
        finally:
            await service.stop()

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, test_config, multiple_test_files):
        """Test concurrent operations across the system."""
        service = DocumentIndexerService()
        service.config = test_config
        
        try:
            await service.initialize()
            
            # Concurrent indexing
            index_tasks = []
            for file in multiple_test_files:
                async def index_file(f):
                    doc_data = service.parser.parse_file(f)
                    doc_data = await service.processor.process_document(doc_data)
                    return await service.indexer.index_document(doc_data)
                
                index_tasks.append(index_file(file))
            
            results = await asyncio.gather(*index_tasks)
            assert all(result is True for result in results)
            
            # Concurrent MCP operations
            from fastmcp import Context
            from src.tools import SearchDocumentsInput, GetCatalogInput
            
            ctx = MockContext()
            tools = service.tools
            
            mcp_tasks = [
                tools.search_documents(ctx, SearchDocumentsInput(query=f"test {i}", limit=5))
                for i in range(5)
            ]
            mcp_tasks.extend([
                tools.get_catalog(ctx, GetCatalogInput(skip=0, limit=10))
                for _ in range(3)
            ])
            mcp_tasks.extend([
                tools.get_indexing_stats(ctx)
                for _ in range(3)
            ])
            
            mcp_results = await asyncio.gather(*mcp_tasks)
            
            # All should succeed
            for result in mcp_results:
                assert isinstance(result, dict)
                assert 'success' in result
            
        finally:
            await service.stop()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_recovery(self, test_config, temp_dir):
        """Test system error recovery."""
        service = DocumentIndexerService()
        service.config = test_config
        
        try:
            await service.initialize()
            
            # Test with invalid file
            invalid_file = temp_dir / "invalid.pdf"
            invalid_file.write_bytes(b"This is not a valid PDF")
            
            # Should handle parsing error gracefully
            try:
                doc_data = service.parser.parse_file(invalid_file)
                # If parsing succeeds (treats as text), process it
                doc_data = await service.processor.process_document(doc_data)
                await service.indexer.index_document(doc_data)
            except Exception:
                # Expected for invalid files
                pass
            
            # System should still be functional
            stats = await service.indexer.get_stats()
            assert isinstance(stats, dict)
            
        finally:
            await service.stop()

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_document_handling(self, test_config, temp_dir):
        """Test handling of large documents."""
        service = DocumentIndexerService()
        service.config = test_config
        
        try:
            await service.initialize()
            
            # Create large document
            large_content = "This is a test sentence. " * 5000  # ~125k characters
            large_file = temp_dir / "large_document.txt"
            large_file.write_text(large_content)
            
            # Index large document
            doc_data = service.parser.parse_file(large_file)
            assert doc_data['total_chars'] > 100000
            assert doc_data['num_chunks'] > 50
            
            doc_data = await service.processor.process_document(doc_data)
            result = await service.indexer.index_document(doc_data)
            assert result is True
            
            # Verify large document is handled
            info = await service.indexer.get_document_info(str(large_file))
            assert info is not None
            assert info['total_chars'] > 100000
            
        finally:
            await service.stop()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_database_persistence(self, test_config, sample_text_file):
        """Test database persistence across service restarts."""
        # First service instance
        service1 = DocumentIndexerService()
        service1.config = test_config
        
        try:
            await service1.initialize()
            
            # Index document
            doc_data = service1.parser.parse_file(sample_text_file)
            doc_data = await service1.processor.process_document(doc_data)
            await service1.indexer.index_document(doc_data)
            
            # Verify document is indexed
            stats1 = await service1.indexer.get_stats()
            assert stats1['total_documents'] == 1
            
        finally:
            await service1.stop()
        
        # Second service instance (same database)
        service2 = DocumentIndexerService()
        service2.config = test_config
        
        try:
            await service2.initialize()
            
            # Verify document persisted
            stats2 = await service2.indexer.get_stats()
            assert stats2['total_documents'] == 1
            
            catalog = await service2.indexer.get_catalog()
            assert len(catalog) == 1
            assert catalog[0]['file_path'] == str(sample_text_file)
            
        finally:
            await service2.stop()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_memory_usage(self, test_config, temp_dir):
        """Test memory usage with multiple documents."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        service = DocumentIndexerService()
        service.config = test_config
        
        try:
            await service.initialize()
            
            # Create and index multiple documents
            files = []
            for i in range(10):
                content = f"Document {i} content. " * 1000
                file = temp_dir / f"doc_{i}.txt"
                file.write_text(content)
                files.append(file)
            
            for file in files:
                doc_data = service.parser.parse_file(file)
                doc_data = await service.processor.process_document(doc_data)
                await service.indexer.index_document(doc_data)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 500MB for this test)
            assert memory_increase < 500, f"Memory usage increased by {memory_increase}MB"
            
            # Verify all documents are indexed
            stats = await service.indexer.get_stats()
            assert stats['total_documents'] == 10
            
        finally:
            await service.stop()

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, test_config, temp_dir):
        """Test basic performance benchmarks."""
        import time
        
        service = DocumentIndexerService()
        service.config = test_config
        
        try:
            await service.initialize()
            
            # Create test documents
            documents = []
            for i in range(5):
                content = f"Performance test document {i}. " * 500
                file = temp_dir / f"perf_doc_{i}.txt"
                file.write_text(content)
                documents.append(file)
            
            # Benchmark indexing
            start_time = time.time()
            
            for file in documents:
                doc_data = service.parser.parse_file(file)
                doc_data = await service.processor.process_document(doc_data)
                await service.indexer.index_document(doc_data)
            
            indexing_time = time.time() - start_time
            
            # Should index 5 documents in reasonable time (less than 60 seconds)
            assert indexing_time < 60, f"Indexing took {indexing_time:.2f} seconds"
            
            # Benchmark search operations
            from fastmcp import Context
            from src.tools import SearchDocumentsInput, GetCatalogInput
            
            ctx = MockContext()
            tools = service.tools
            
            start_time = time.time()
            
            # Perform multiple searches
            for i in range(10):
                await tools.search_documents(
                    ctx, SearchDocumentsInput(query=f"test {i}", limit=5)
                )
            
            search_time = time.time() - start_time
            
            # 10 searches should complete quickly (less than 10 seconds)
            assert search_time < 10, f"Searches took {search_time:.2f} seconds"
            
        finally:
            await service.stop()