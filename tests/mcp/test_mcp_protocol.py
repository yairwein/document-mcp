"""Tests for MCP protocol compliance and parameter handling."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from fastmcp import FastMCP
from pydantic import ValidationError

from src.tools import (
    SearchDocumentsInput, 
    GetCatalogInput, 
    GetDocumentInfoInput, 
    ReindexDocumentInput,
    DocumentTools
)


class MockContext:
    """Mock Context for testing."""
    pass


class TestMCPProtocolCompliance:
    """Test MCP protocol compliance and parameter handling."""

    @pytest.mark.mcp
    def test_pydantic_model_validation(self):
        """Test Pydantic model validation for MCP inputs."""
        # Valid SearchDocumentsInput
        valid_search = SearchDocumentsInput(query="test", limit=5, search_type="documents")
        assert valid_search.query == "test"
        assert valid_search.limit == 5
        assert valid_search.search_type == "documents"
        
        # Valid with defaults
        minimal_search = SearchDocumentsInput(query="test")
        assert minimal_search.limit == 10  # default
        assert minimal_search.search_type == "documents"  # default
        
        # Invalid search type should raise ValidationError
        with pytest.raises(ValidationError):
            SearchDocumentsInput(query="test", search_type="invalid")

    @pytest.mark.mcp
    def test_catalog_input_validation(self):
        """Test catalog input validation."""
        # Valid inputs
        valid_catalog = GetCatalogInput(skip=0, limit=10)
        assert valid_catalog.skip == 0
        assert valid_catalog.limit == 10
        
        # With defaults
        default_catalog = GetCatalogInput()
        assert default_catalog.skip == 0
        assert default_catalog.limit == 100

    @pytest.mark.mcp
    def test_document_info_input_validation(self):
        """Test document info input validation."""
        valid_info = GetDocumentInfoInput(file_path="/path/to/file.txt")
        assert valid_info.file_path == "/path/to/file.txt"
        
        # Missing required field
        with pytest.raises(ValidationError):
            GetDocumentInfoInput()

    @pytest.mark.mcp
    def test_reindex_input_validation(self):
        """Test reindex input validation."""
        valid_reindex = ReindexDocumentInput(file_path="/path/to/file.pdf")
        assert valid_reindex.file_path == "/path/to/file.pdf"

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_tool_call_with_valid_params(self, document_tools):
        """Test MCP tool calls with valid parameters."""
        ctx = MockContext()
        
        # Test search with valid params
        search_params = SearchDocumentsInput(query="test document", limit=5)
        result = await document_tools.search_documents(ctx, search_params)
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert isinstance(result['success'], bool)

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_response_schema_compliance(self, document_tools):
        """Test that MCP responses follow consistent schema."""
        ctx = MockContext()
        
        # Test all tools return consistent response structure
        search_result = await document_tools.search_documents(
            ctx, SearchDocumentsInput(query="test")
        )
        catalog_result = await document_tools.get_catalog(
            ctx, GetCatalogInput()
        )
        stats_result = await document_tools.get_indexing_stats(ctx)
        
        # All should have 'success' field
        for result in [search_result, catalog_result, stats_result]:
            assert 'success' in result
            assert isinstance(result['success'], bool)
            
            # If successful, should have expected fields
            if result['success']:
                # Each tool has specific fields but all should be JSON serializable
                json.dumps(result)  # Should not raise

    @pytest.mark.mcp
    def test_json_schema_generation(self):
        """Test that Pydantic models can generate JSON schemas for MCP."""
        # Test that input models can generate schemas
        search_schema = SearchDocumentsInput.model_json_schema()
        assert 'properties' in search_schema
        assert 'query' in search_schema['properties']
        assert 'limit' in search_schema['properties']
        
        catalog_schema = GetCatalogInput.model_json_schema()
        assert 'properties' in catalog_schema
        assert 'skip' in catalog_schema['properties']
        assert 'limit' in catalog_schema['properties']

    @pytest.mark.mcp
    def test_parameter_serialization_deserialization(self):
        """Test parameter serialization/deserialization for MCP protocol."""
        # Create input object
        search_input = SearchDocumentsInput(query="test", limit=5, search_type="chunks")
        
        # Serialize to dict (as MCP would receive)
        input_dict = search_input.model_dump()
        assert input_dict == {
            'query': 'test',
            'limit': 5,
            'search_type': 'chunks'
        }
        
        # Deserialize back (as FastMCP would do)
        recreated_input = SearchDocumentsInput.model_validate(input_dict)
        assert recreated_input.query == search_input.query
        assert recreated_input.limit == search_input.limit
        assert recreated_input.search_type == search_input.search_type

    @pytest.mark.mcp
    def test_json_serialization_edge_cases(self):
        """Test JSON serialization of edge cases."""
        # Test with special characters
        search_input = SearchDocumentsInput(
            query='test "quoted" text with special chars: àáâã',
            limit=10
        )
        
        # Should serialize/deserialize correctly
        json_str = json.dumps(search_input.model_dump())
        data = json.loads(json_str)
        recreated = SearchDocumentsInput.model_validate(data)
        
        assert recreated.query == search_input.query

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_error_response_format(self, document_tools):
        """Test standardized error response format."""
        ctx = MockContext()
        
        # Test error response for nonexistent document
        error_result = await document_tools.get_document_info(
            ctx, GetDocumentInfoInput(file_path="/nonexistent/file.txt")
        )
        
        # Error responses should follow standard format
        assert isinstance(error_result, dict)
        assert error_result['success'] is False
        assert 'error' in error_result
        assert isinstance(error_result['error'], str)
        assert len(error_result['error']) > 0

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_parameter_type_coercion(self):
        """Test parameter type coercion in MCP inputs."""
        # Test that string numbers are handled
        input_data = {
            'query': 'test',
            'limit': '5',  # String instead of int
            'search_type': 'documents'
        }
        
        # Pydantic should coerce string to int
        search_input = SearchDocumentsInput.model_validate(input_data)
        assert isinstance(search_input.limit, int)
        assert search_input.limit == 5

    @pytest.mark.mcp
    def test_mcp_default_value_handling(self):
        """Test default value handling in MCP inputs."""
        # Minimal input should use defaults
        minimal_data = {'query': 'test'}
        search_input = SearchDocumentsInput.model_validate(minimal_data)
        
        assert search_input.query == 'test'
        assert search_input.limit == 10  # default
        assert search_input.search_type == 'documents'  # default

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_tool_context_isolation(self, document_tools):
        """Test that MCP tool contexts are properly isolated."""
        ctx1 = MockContext()
        ctx2 = MockContext()
        
        # Calls with different contexts should not interfere
        result1 = await document_tools.get_indexing_stats(ctx1)
        result2 = await document_tools.get_indexing_stats(ctx2)
        
        # Results should be identical (stateless tools)
        assert result1 == result2

    @pytest.mark.mcp
    def test_input_validation_edge_cases(self):
        """Test input validation edge cases."""
        # Empty query
        with pytest.raises(ValidationError):
            SearchDocumentsInput(query="")
        
        # Very long query (should be handled gracefully)
        long_query = "test " * 1000
        search_input = SearchDocumentsInput(query=long_query)
        assert len(search_input.query) > 1000
        
        # Negative limit
        with pytest.raises(ValidationError):
            GetCatalogInput(skip=0, limit=-1)
        
        # Very large limit (should be handled)
        large_catalog = GetCatalogInput(skip=0, limit=10000)
        assert large_catalog.limit == 10000

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_response_consistency(self, document_tools, document_parser, document_processor, sample_text_file):
        """Test that MCP responses are consistent across calls."""
        # Index a document
        doc_data = document_parser.parse_file(sample_text_file)
        doc_data = await document_processor.process_document(doc_data)
        await document_tools.indexer.index_document(doc_data)
        
        ctx = MockContext()
        
        # Make multiple identical calls
        results = []
        for _ in range(3):
            result = await document_tools.get_indexing_stats(ctx)
            results.append(result)
        
        # All results should be identical
        for i in range(1, len(results)):
            assert results[i] == results[0]

    @pytest.mark.mcp
    def test_pydantic_field_descriptions(self):
        """Test that Pydantic fields have proper descriptions for MCP schema."""
        schema = SearchDocumentsInput.model_json_schema()
        
        # Check field descriptions exist
        query_field = schema['properties']['query']
        assert 'description' in query_field
        assert len(query_field['description']) > 0
        
        limit_field = schema['properties']['limit']
        assert 'description' in limit_field

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_large_parameter_handling(self, document_tools):
        """Test handling of large parameters in MCP calls."""
        ctx = MockContext()
        
        # Test with very large query
        large_query = "search term " * 100
        search_input = SearchDocumentsInput(query=large_query, limit=5)
        
        result = await document_tools.search_documents(ctx, search_input)
        
        # Should handle large query gracefully
        assert isinstance(result, dict)
        assert 'success' in result

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_concurrent_parameter_validation(self):
        """Test parameter validation under concurrent access."""
        import asyncio
        
        # Create multiple validation tasks concurrently
        tasks = []
        for i in range(10):
            task = asyncio.create_task(
                asyncio.to_thread(
                    SearchDocumentsInput.model_validate,
                    {'query': f'test {i}', 'limit': i + 1}
                )
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All validations should succeed
        for i, result in enumerate(results):
            assert result.query == f'test {i}'
            assert result.limit == i + 1

    @pytest.mark.mcp
    def test_mcp_input_model_inheritance(self):
        """Test that input models properly inherit from BaseModel."""
        from pydantic import BaseModel
        
        assert issubclass(SearchDocumentsInput, BaseModel)
        assert issubclass(GetCatalogInput, BaseModel)
        assert issubclass(GetDocumentInfoInput, BaseModel)
        assert issubclass(ReindexDocumentInput, BaseModel)

    @pytest.mark.mcp
    @pytest.mark.asyncio
    async def test_mcp_tool_exception_handling(self, temp_dir):
        """Test MCP tool exception handling and error responses."""
        from src.indexer import DocumentIndexer
        from src.parser import DocumentParser  
        from src.llm import LocalLLM, DocumentProcessor
        
        # Create tools that will likely fail
        indexer = DocumentIndexer(temp_dir / "invalid", "invalid-model")
        parser = DocumentParser()
        llm = LocalLLM("invalid", "http://invalid:11434")
        await llm.initialize()
        processor = DocumentProcessor(llm)
        
        tools = DocumentTools(indexer, parser, processor)
        ctx = MockContext()
        
        # All tools should handle internal exceptions gracefully
        search_result = await tools.search_documents(
            ctx, SearchDocumentsInput(query="test")
        )
        assert isinstance(search_result, dict)
        assert 'success' in search_result
        
        catalog_result = await tools.get_catalog(ctx, GetCatalogInput())
        assert isinstance(catalog_result, dict)
        assert 'success' in catalog_result
        
        await llm.close()