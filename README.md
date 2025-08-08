# MCP Document Indexer

A Python-based MCP (Model Context Protocol) server for local document indexing and search using LanceDB vector database and local LLMs.

## Features

- **Real-time Document Monitoring**: Automatically indexes new and modified documents in configured folders
- **Multi-format Support**: Handles PDF, Word (docx/doc), text, Markdown, and RTF files
- **Local LLM Integration**: Uses Ollama for document summarization and keyword extraction. Nothing ever leaves your computer
- **Vector Search**: Semantic search using LanceDB and sentence transformers
- **MCP Integration**: Exposes search and catalog tools via Model Context Protocol
- **Incremental Indexing**: Only processes changed files to save resources
- **Performance Optimized**: Designed for decent performance on standard laptops (e.g. M1/M2 MacBook)

## Installation

### Prerequisites

1. **Python 3.9+** installed
2. **uv** package manager:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Ollama** (for local LLM):
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (e.g., llama3.2)
ollama pull llama3.2:3b
```

### Install MCP Document Indexer

```bash
# Clone the repository
git clone https://github.com/yairwein/mcp-doc-indexer.git
cd mcp-doc-indexer

# Install with uv
uv sync

# Or install as a package
uv add mcp-doc-indexer
```

## Configuration

Configure the indexer using environment variables or a `.env` file:

```bash
# Folders to monitor (comma-separated)
WATCH_FOLDERS="/Users/me/Documents,/Users/me/Research"

# LanceDB storage path
LANCEDB_PATH="./vector_index"

# Ollama model for summarization
LLM_MODEL="llama3.2:3b"

# Text chunking settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Embedding model (sentence-transformers)
EMBEDDING_MODEL="all-MiniLM-L6-v2"

# File types to index
FILE_EXTENSIONS=".pdf,.docx,.doc,.txt,.md,.rtf"

# Maximum file size in MB
MAX_FILE_SIZE_MB=100

# Ollama API URL
OLLAMA_BASE_URL="http://localhost:11434"
```

## Usage

### Run as Standalone Service

```bash
# Set environment variables
export WATCH_FOLDERS="/path/to/documents"
export LANCEDB_PATH="./my_index"

# Run the indexer
uv run python -m src.main
```

### Integrate with Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "doc-indexer": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.main"],
      "cwd": "/path/to/mcp-doc-indexer",
      "env": {
        "WATCH_FOLDERS": "/Users/me/Documents,/Users/me/Research",
        "LANCEDB_PATH": "/Users/me/.mcp-doc-index",
        "LLM_MODEL": "llama3.2:3b"
      }
    }
  }
}
```

## MCP Tools

The indexer exposes the following tools via MCP:

### `search_documents`
Search for documents using natural language queries.
- **Parameters**:
  - `query`: Search query text
  - `limit`: Maximum number of results (default: 10)
  - `search_type`: "documents" or "chunks"

### `get_catalog`
List all indexed documents with summaries.
- **Parameters**:
  - `skip`: Number of documents to skip (default: 0)
  - `limit`: Maximum documents to return (default: 100)

### `get_document_info`
Get detailed information about a specific document.
- **Parameters**:
  - `file_path`: Path to the document

### `reindex_document`
Force reindexing of a specific document.
- **Parameters**:
  - `file_path`: Path to the document to reindex

### `get_indexing_stats`
Get current indexing statistics.

## Example Usage in Claude

Once configured, you can use the indexer in Claude:

```
"Search my documents for information about machine learning"
"Show me all PDFs I've indexed"
"What documents mention Python programming?"
"Get details about /Users/me/Documents/report.pdf"
"Reindex the latest version of my thesis"
```

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  File Monitor   │────▶│   Document   │────▶│  Local LLM  │
│   (Watchdog)    │     │    Parser    │     │  (Ollama)   │
└─────────────────┘     └──────────────┘     └─────────────┘
                               │                      │
                               ▼                      ▼
                        ┌──────────────┐     ┌─────────────┐
                        │   LanceDB    │◀────│  Embeddings │
                        │   Storage    │     │  (ST Model) │
                        └──────────────┘     └─────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  FastMCP     │
                        │   Server     │
                        └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │    Claude    │
                        │   Desktop    │
                        └──────────────┘
```

## File Processing Pipeline

1. **File Detection**: Watchdog monitors configured folders for changes
2. **Document Parsing**: Extracts text from PDF, Word, and text files
3. **Text Chunking**: Splits documents into overlapping chunks for better retrieval
4. **LLM Processing**: Generates summaries and extracts keywords using Ollama
5. **Embedding Generation**: Creates vector embeddings using sentence transformers
6. **Vector Storage**: Stores documents and chunks in LanceDB
7. **MCP Exposure**: Makes search and catalog tools available via MCP

## Performance Considerations

- **Incremental Indexing**: Only changed files are reprocessed
- **Async Processing**: Parallel processing of multiple documents
- **Batch Operations**: Efficient batch indexing for multiple files
- **Debouncing**: Prevents duplicate processing of rapidly changing files
- **Size Limits**: Configurable maximum file size to prevent memory issues

## Troubleshooting

### Ollama Not Available
If Ollama is not running or the model isn't available, the indexer falls back to simple text extraction without summarization.

```bash
# Check Ollama status
ollama list

# Pull required model
ollama pull llama3.2:3b
```

### Permission Issues
Ensure the indexer has read access to monitored folders:
```bash
chmod -R 755 /path/to/documents
```

### Memory Usage
For large document collections, consider:
- Reducing `CHUNK_SIZE` to create smaller chunks
- Limiting `MAX_FILE_SIZE_MB` to skip very large files
- Using a smaller embedding model

## Development

### Running Tests
```bash
uv run pytest tests/
```

### Code Formatting
```bash
uv run black src/
uv run ruff src/
```

### Building Package
```bash
uv build
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues or questions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review logs in the console output
