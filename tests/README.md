# Test Suite for MCP Document Indexer

This directory contains a comprehensive test suite for the MCP Document Indexer project, including unit tests, integration tests, and MCP-specific functionality tests.

## Test Structure

```
tests/
├── conftest.py                 # Test configuration and fixtures
├── unit/                       # Unit tests
│   ├── test_parser.py         # Document parser tests
│   ├── test_llm.py            # LLM and processor tests
│   └── test_indexer.py        # Document indexer tests
├── integration/                # Integration tests
│   └── test_tools.py          # MCP tools integration tests
├── mcp/                        # MCP-specific tests
│   ├── test_mcp_server.py     # MCP server functionality
│   └── test_mcp_protocol.py   # MCP protocol compliance
└── test_end_to_end.py         # End-to-end integration tests
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Parser Tests**: Test document parsing, chunking, and metadata extraction
- **LLM Tests**: Test LLM integration, document processing, and summarization  
- **Indexer Tests**: Test vector indexing, search, and database operations

### Integration Tests (`tests/integration/`)
- **Tools Tests**: Test MCP tools with real components integrated together
- Test document lifecycle from parsing through indexing to search

### MCP Tests (`tests/mcp/`)
- **Server Tests**: Test MCP server setup, tool registration, and lifecycle
- **Protocol Tests**: Test MCP protocol compliance, parameter validation, and error handling

### End-to-End Tests (`test_end_to_end.py`)
- Complete workflow tests from document ingestion to search
- Performance and memory usage benchmarks
- Concurrent operation testing
- Database persistence testing

## Running Tests

### Quick Start
```bash
# Run all tests
make test

# Run specific test categories
make test-unit          # Unit tests only
make test-integration   # Integration tests only  
make test-mcp          # MCP tests only

# Run fast tests (skip slow external service tests)
make test-fast

# Run with coverage
make test-all
```

### Detailed Test Commands

```bash
# Unit tests for specific components
uv run pytest tests/unit/test_parser.py -v
uv run pytest tests/unit/test_llm.py -v
uv run pytest tests/unit/test_indexer.py -v

# Integration tests
uv run pytest tests/integration/ -v

# MCP functionality tests
uv run pytest tests/mcp/ -v

# End-to-end tests
uv run pytest tests/test_end_to_end.py -v

# Run tests with specific markers
uv run pytest -m "unit" -v        # Only unit tests
uv run pytest -m "integration" -v # Only integration tests
uv run pytest -m "mcp" -v        # Only MCP tests
uv run pytest -m "slow" -v       # Only slow tests
uv run pytest -m "not slow" -v   # Skip slow tests
```

### Test Markers

Tests are organized with pytest markers:

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (moderate speed)
- `@pytest.mark.mcp` - MCP-specific functionality tests
- `@pytest.mark.slow` - Tests requiring external services (Ollama, etc.)

## Test Configuration

### Environment Setup

Tests use temporary directories and isolated configurations to avoid interfering with production data:

- Each test gets a fresh temporary directory
- Test databases are created in temp locations
- LLM tests can run with mock responses when Ollama is unavailable

### Fixtures

Key fixtures provided in `conftest.py`:

- `test_config` - Isolated test configuration
- `document_parser` - Document parser instance
- `llm` - LLM instance (with fallback for offline testing)
- `document_processor` - Document processor with LLM
- `document_indexer` - Vector database indexer
- `document_tools` - MCP tools with all components
- `sample_text_file` - Sample text document
- `sample_legal_file` - Sample legal document (NDA)
- `multiple_test_files` - Collection of test documents

### Dependencies

Some tests require external services:

- **Ollama** - For LLM functionality tests (marked as `slow`)
- **Sentence Transformers** - For embedding generation
- **LanceDB** - For vector storage (uses temp databases)

Tests gracefully handle missing dependencies with appropriate fallbacks or skips.

## MCP-Specific Testing

### Parameter Validation Testing

Tests ensure MCP tools properly validate input parameters:

```python
# Valid parameters
search_input = SearchDocumentsInput(query="test", limit=5)

# Invalid parameters should raise ValidationError
with pytest.raises(ValidationError):
    SearchDocumentsInput(query="", limit=-1)
```

### Protocol Compliance Testing

Tests verify MCP protocol compliance:

- JSON serialization/deserialization
- Consistent response formats
- Error handling and reporting
- Parameter type coercion

### Server Integration Testing

Tests verify MCP server functionality:

- Tool registration and discovery
- Context handling
- Concurrent request handling
- Resource cleanup

## Performance Testing

### Benchmarks

End-to-end tests include performance benchmarks:

- Document indexing speed
- Search operation latency
- Memory usage monitoring
- Concurrent operation handling

### Load Testing

Tests verify system behavior under load:

- Multiple concurrent indexing operations
- Concurrent MCP tool requests  
- Large document handling
- Database performance

## Debugging Tests

### Verbose Output
```bash
# Run with verbose output and no capture
uv run pytest -v -s --tb=long

# Debug specific test
uv run pytest tests/unit/test_parser.py::TestDocumentParser::test_parse_text_file -v -s
```

### Test Isolation
```bash
# Run single test method
uv run pytest tests/mcp/test_mcp_server.py::TestMCPServer::test_mcp_server_setup -v

# Run tests matching pattern
uv run pytest -k "legal" -v
```

### Failure Analysis
```bash
# Re-run only failed tests
make test-failed

# Show local variables on failure
uv run pytest --tb=long --showlocals
```

## Contributing Tests

### Adding New Tests

1. **Unit Tests**: Add to appropriate file in `tests/unit/`
2. **Integration Tests**: Add to `tests/integration/`  
3. **MCP Tests**: Add to `tests/mcp/`
4. **End-to-End Tests**: Add to `tests/test_end_to_end.py`

### Test Guidelines

1. **Use appropriate markers** (`@pytest.mark.unit`, etc.)
2. **Use descriptive test names** that explain what is being tested
3. **Test both success and failure cases**
4. **Use fixtures** for common setup
5. **Mock external dependencies** in unit tests
6. **Test error handling** and edge cases
7. **Keep tests isolated** and independent

### Test Naming Convention

```python
class TestComponentName:
    def test_specific_functionality(self):
        """Test description of what this test verifies."""
        pass
    
    def test_error_case_description(self):
        """Test specific error condition handling."""
        pass
```

## Test Data

Test files and content are generated dynamically to avoid committing large test files:

- Text documents with various content types
- Legal documents (NDAs, contracts)
- Technical documentation
- Large documents for performance testing

All test data is created in temporary directories and cleaned up automatically.

## Continuous Integration

The test suite is designed for CI environments:

- Fast test subset for quick feedback
- Comprehensive test suite for full validation
- Proper timeout handling for slow tests
- Clear failure reporting and debugging information

Use `make ci-test` for CI-optimized test runs with failure limits and concise output.