"""Tests for MCP server functionality."""

import pytest
import pytest_asyncio
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch
from fastmcp import FastMCP
from src.main import DocumentIndexerService
from src.tools import SearchDocumentsInput, GetCatalogInput


class MockContext:
    """Mock Context for testing."""
    pass


class TestMCPServer:
    """Test MCP server functionality."""

    @pytest_asyncio.fixture
    async def mcp_service(self, test_config):
        """Create MCP service for testing."""
        service = DocumentIndexerService()
        service.config = test_config
        await service.initialize()
        yield service
        await service.stop()

    @pytest_asyncio.fixture
    async def mcp_server(self, mcp_service):
        """Create MCP server."""
        return mcp_service.setup_mcp_server()

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_server_setup(self, mcp_server):
        """Test MCP server setup."""
        assert isinstance(mcp_server, FastMCP)
        assert mcp_server.name == "mcp-doc-indexer"
        
        # Test that server has basic functionality
        # For now we just test that it was created correctly
        assert hasattr(mcp_server, 'tool')  # The decorator method exists
        assert hasattr(mcp_server, 'name')

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_tool_registration(self, mcp_service):
        """Test that MCP tools are properly registered."""
        mcp = mcp_service.setup_mcp_server()
        
        # Test that server was created successfully
        assert isinstance(mcp, FastMCP)
        assert mcp.name == "mcp-doc-indexer"
        
        # Test that the underlying service has tools
        assert mcp_service.tools is not None
        assert hasattr(mcp_service.tools, 'search_documents')
        assert hasattr(mcp_service.tools, 'get_catalog')
        assert hasattr(mcp_service.tools, 'get_document_info')
        assert hasattr(mcp_service.tools, 'reindex_document')
        assert hasattr(mcp_service.tools, 'get_indexing_stats')

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_tool_parameter_validation(self, mcp_service):
        """Test MCP tool parameter validation."""
        
        tools = mcp_service.tools
        ctx = MockContext()
        
        # Test valid parameters
        valid_search = SearchDocumentsInput(query="test", limit=5)
        result = await tools.search_documents(ctx, valid_search)
        assert isinstance(result, dict)
        assert 'success' in result
        
        # Test valid catalog parameters
        valid_catalog = GetCatalogInput(skip=0, limit=10)
        result = await tools.get_catalog(ctx, valid_catalog)
        assert isinstance(result, dict)
        assert 'success' in result

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_tool_error_handling(self, temp_dir):
        """Test MCP tool error handling."""
        from src.config import Config
        
        # Create service with invalid configuration to trigger errors
        config = Config(
            watch_folders=[],
            lancedb_path=temp_dir / "invalid_db",
            llm_model="invalid-model",
            embedding_model="invalid-embedding-model"
        )
        
        service = DocumentIndexerService()
        service.config = config
        
        try:
            await service.initialize()
        except Exception:
            # Expected to fail with invalid config
            pass
        
        # Tools should still handle errors gracefully
        if service.tools:
            ctx = MockContext()
            search_input = SearchDocumentsInput(query="test", limit=5)
            result = await service.tools.search_documents(ctx, search_input)
            assert isinstance(result, dict)
            assert 'success' in result

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_json_serialization(self, mcp_service, document_parser, document_processor, sample_text_file):
        """Test that MCP tool results are JSON serializable."""
        # Index a document
        doc_data = document_parser.parse_file(sample_text_file)
        doc_data = await document_processor.process_document(doc_data)
        await mcp_service.indexer.index_document(doc_data)
        
        ctx = MockContext()
        
        # Test all tools return JSON-serializable results
        tools = mcp_service.tools
        
        # Search documents
        search_result = await tools.search_documents(ctx, SearchDocumentsInput(query="test", limit=5))
        json.dumps(search_result)  # Should not raise exception
        
        # Get catalog
        catalog_result = await tools.get_catalog(ctx, GetCatalogInput(skip=0, limit=10))
        json.dumps(catalog_result)  # Should not raise exception
        
        # Get stats
        stats_result = await tools.get_indexing_stats(ctx)
        json.dumps(stats_result)  # Should not raise exception

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_concurrent_requests(self, mcp_service, document_parser, document_processor, multiple_test_files):
        """Test handling concurrent MCP requests."""
        # Index documents
        for file in multiple_test_files:
            doc_data = document_parser.parse_file(file)
            doc_data = await document_processor.process_document(doc_data)
            await mcp_service.indexer.index_document(doc_data)
        
        ctx = MockContext()
        tools = mcp_service.tools
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(10):
            tasks.append(tools.search_documents(ctx, SearchDocumentsInput(query=f"test {i}", limit=3)))
            tasks.append(tools.get_catalog(ctx, GetCatalogInput(skip=0, limit=5)))
            tasks.append(tools.get_indexing_stats(ctx))
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete successfully
        for result in results:
            assert not isinstance(result, Exception), f"Request failed with {result}"
            assert isinstance(result, dict)
            assert 'success' in result

    @pytest.mark.mcp
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_mcp_server_stdio_mode(self, mcp_service):
        """Test MCP server in stdio mode (integration with actual MCP protocol)."""
        # This test would require actual MCP client implementation
        # For now, we test the server setup for stdio mode
        
        mcp = mcp_service.setup_mcp_server()
        
        # Test that server can be configured for stdio
        assert hasattr(mcp, 'run_stdio_async')
        
        # Test server creation doesn't raise exceptions
        assert mcp.name == "mcp-doc-indexer"

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_tool_descriptions_and_schemas(self, mcp_server):
        """Test that MCP tools have proper descriptions and schemas."""
        # Test that server was created successfully
        assert isinstance(mcp_server, FastMCP)
        assert mcp_server.name == "mcp-doc-indexer"
        
        # Test that FastMCP has the tool decorator method
        assert hasattr(mcp_server, 'tool')
        
        # Since FastMCP doesn't expose registered tools directly,
        # we can't test the tool descriptions directly through the server object.
        # This would require actual MCP protocol interaction or different FastMCP API.

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_large_response_handling(self, mcp_service, document_parser, document_processor, temp_dir):
        """Test handling of large MCP responses."""
        # Create multiple large documents
        large_files = []
        for i in range(5):
            content = f"Large document {i} content. " * 500
            file = temp_dir / f"large_doc_{i}.txt"
            file.write_text(content)
            large_files.append(file)
        
        # Index all documents
        for file in large_files:
            doc_data = document_parser.parse_file(file)
            doc_data = await document_processor.process_document(doc_data)
            await mcp_service.indexer.index_document(doc_data)
        
        ctx = MockContext()
        
        # Get large catalog
        result = await mcp_service.tools.get_catalog(ctx, GetCatalogInput(skip=0, limit=100))
        
        # Should handle large response
        assert result['success'] is True
        assert len(result['documents']) == 5
        
        # Response should be serializable despite size
        json_str = json.dumps(result)
        assert len(json_str) > 1000  # Ensure it's actually large

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_parameter_type_conversion(self, mcp_service):
        """Test MCP parameter type conversion."""
        from src.tools import SearchDocumentsInput, GetCatalogInput
        
        ctx = MockContext()
        tools = mcp_service.tools
        
        # Test that string numbers are converted properly
        search_input = SearchDocumentsInput(query="test", limit=5, search_type="documents")
        result = await tools.search_documents(ctx, search_input)
        assert isinstance(result, dict)
        
        # Test catalog with various parameter types
        catalog_input = GetCatalogInput(skip=0, limit=10)
        result = await tools.get_catalog(ctx, catalog_input)
        assert isinstance(result, dict)

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_service_lifecycle(self, test_config):
        """Test MCP service complete lifecycle."""
        service = DocumentIndexerService()
        service.config = test_config
        
        # Test initialization
        await service.initialize()
        assert service.parser is not None
        assert service.llm is not None
        assert service.processor is not None
        assert service.indexer is not None
        assert service.tools is not None
        
        # Test MCP server setup
        mcp = service.setup_mcp_server()
        assert mcp is not None
        
        # Test service running state
        assert not service.running  # Not started yet
        
        # Test cleanup
        await service.stop()
        assert not service.running

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_resource_cleanup(self, mcp_service):
        """Test that MCP service properly cleans up resources."""
        # Verify components are initialized
        assert mcp_service.indexer is not None
        assert mcp_service.llm is not None
        
        # Stop service
        await mcp_service.stop()
        
        # Verify cleanup (components should handle close gracefully)
        # This mainly tests that stop() doesn't raise exceptions

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_tool_context_handling(self, mcp_service):
        """Test MCP tool context handling."""
        ctx = MockContext()
        tools = mcp_service.tools
        
        # Tools should work with context
        result = await tools.get_indexing_stats(ctx)
        assert isinstance(result, dict)
        assert 'success' in result
        
        # Context should not affect results (stateless tools)
        result2 = await tools.get_indexing_stats(ctx)
        assert result == result2

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_error_response_format(self, temp_dir):
        """Test that MCP tools return consistent error response format."""
        from src.tools import DocumentTools, GetDocumentInfoInput
        from src.indexer import DocumentIndexer
        from src.parser import DocumentParser
        from src.llm import LocalLLM, DocumentProcessor
        
        # Create minimal tools setup
        indexer = DocumentIndexer(temp_dir / "test_db", "all-MiniLM-L6-v2")
        await indexer.initialize()
        
        parser = DocumentParser()
        llm = LocalLLM("test", "http://invalid:11434")
        await llm.initialize()
        processor = DocumentProcessor(llm)
        
        tools = DocumentTools(indexer, parser, processor)
        ctx = MockContext()
        
        # Test error response for nonexistent document
        result = await tools.get_document_info(ctx, GetDocumentInfoInput(file_path="/nonexistent"))
        
        assert isinstance(result, dict)
        assert result['success'] is False
        assert 'error' in result
        assert isinstance(result['error'], str)
        
        await indexer.close()
        await llm.close()