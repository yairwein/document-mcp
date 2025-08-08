"""MCP tool implementations for document indexing."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastmcp import Context
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SearchDocumentsInput(BaseModel):
    """Input for document search."""
    query: str = Field(min_length=1, description="Search query text")
    limit: int = Field(default=10, ge=1, description="Maximum number of results")
    search_type: str = Field(
        default="documents",
        pattern="^(documents|chunks)$",
        description="Search type: 'documents' for document-level or 'chunks' for chunk-level"
    )


class GetCatalogInput(BaseModel):
    """Input for getting catalog."""
    skip: int = Field(default=0, ge=0, description="Number of documents to skip")
    limit: int = Field(default=100, ge=1, description="Maximum number of documents to return")


class GetDocumentInfoInput(BaseModel):
    """Input for getting document info."""
    file_path: str = Field(description="Path to the document file")


class ReindexDocumentInput(BaseModel):
    """Input for reindexing a document."""
    file_path: str = Field(description="Path to the document file to reindex")


class DocumentTools:
    """MCP tools for document operations."""
    
    def __init__(self, indexer, parser, processor):
        self.indexer = indexer
        self.parser = parser
        self.processor = processor
    
    async def search_documents(self, ctx: Context, input: SearchDocumentsInput) -> Dict[str, Any]:
        """
        Search for documents or chunks using semantic search.
        
        This tool searches through indexed documents using natural language queries.
        It can search at the document level (returning whole documents) or chunk level
        (returning specific passages).
        
        Sample queries:
        - "Find documents about machine learning algorithms"
        - "Search for API documentation"  
        - "Show me code related to database connections"
        - "Find text about authentication and security"
        - "Look for configuration files and setup instructions"
        """
        try:
            if input.search_type == "chunks":
                results = await self.indexer.search_chunks(input.query, input.limit)
                
                # Group chunks by document
                docs_chunks = {}
                for chunk in results:
                    file_path = chunk['file_path']
                    if file_path not in docs_chunks:
                        docs_chunks[file_path] = []
                    docs_chunks[file_path].append({
                        'chunk_id': chunk['chunk_id'],
                        'text': chunk['chunk_text'][:500],  # Truncate for response
                        'char_count': chunk['char_count']
                    })
                
                return {
                    "success": True,
                    "query": input.query,
                    "search_type": "chunks",
                    "total_results": len(results),
                    "results": docs_chunks
                }
            else:
                results = await self.indexer.search_documents(input.query, input.limit)
                
                # Format results
                formatted_results = []
                for doc in results:
                    formatted_results.append({
                        'file_path': doc['file_path'],
                        'file_name': doc['file_name'],
                        'summary': doc['summary'],
                        'keywords': doc['keywords'],
                        'file_type': doc['file_type'],
                        'modified_time': doc['modified_time']
                    })
                
                return {
                    "success": True,
                    "query": input.query,
                    "search_type": "documents",
                    "total_results": len(results),
                    "results": formatted_results
                }
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": input.query,
                "search_type": input.search_type,
                "total_results": 0,
                "results": []
            }
    
    async def get_catalog(self, ctx: Context, input: GetCatalogInput) -> Dict[str, Any]:
        """
        Get a list of all indexed documents with their summaries.
        
        Returns a catalog of all documents that have been indexed, including
        their metadata, summaries, and keywords. Useful for browsing the
        document collection.
        
        Sample usage:
        - Browse all indexed documents to see what's available
        - Get an overview of the document collection with metadata
        - Check which file types are indexed and their summaries
        - Find documents by scanning through titles and keywords
        """
        try:
            documents = await self.indexer.get_catalog(input.skip, input.limit)
            
            # Get stats
            stats = await self.indexer.get_stats()
            
            # Format documents
            formatted_docs = []
            for doc in documents:
                formatted_docs.append({
                    'file_path': doc['file_path'],
                    'file_name': doc['file_name'],
                    'summary': doc['summary'][:200] if doc['summary'] else "No summary",
                    'keywords': doc['keywords'][:5] if doc['keywords'] else [],
                    'file_type': doc['file_type'],
                    'file_size': doc['file_size'],
                    'total_chunks': doc['total_chunks'],
                    'indexed_time': doc['indexed_time']
                })
            
            return {
                "success": True,
                "total_documents": stats['total_documents'],
                "returned": len(formatted_docs),
                "skip": input.skip,
                "limit": input.limit,
                "documents": formatted_docs,
                "stats": {
                    "total_chunks": stats['total_chunks'],
                    "total_size_mb": round(stats['total_size_bytes'] / (1024 * 1024), 2),
                    "file_types": stats['file_types']
                }
            }
            
        except Exception as e:
            logger.error(f"Catalog error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_document_info(self, ctx: Context, input: GetDocumentInfoInput) -> Dict[str, Any]:
        """
        Get detailed information about a specific indexed document.
        
        Returns comprehensive information about a document including its
        summary, keywords, chunk count, and indexing metadata.
        
        Sample usage:
        - Get details about "/path/to/my/important-document.pdf"
        - Check metadata for "src/main.py" including chunk count and size
        - View summary and keywords for "docs/README.md"
        - Inspect indexing status and timestamps for any file
        """
        try:
            doc_info = await self.indexer.get_document_info(input.file_path)
            
            if not doc_info:
                return {
                    "success": False,
                    "error": f"Document not found: {input.file_path}"
                }
            
            return {
                "success": True,
                "document": {
                    'file_path': doc_info['file_path'],
                    'file_name': doc_info['file_name'],
                    'file_hash': doc_info['file_hash'],
                    'file_size': doc_info['file_size'],
                    'file_size_mb': round(doc_info['file_size'] / (1024 * 1024), 2),
                    'file_type': doc_info['file_type'],
                    'modified_time': doc_info['modified_time'],
                    'indexed_time': doc_info['indexed_time'],
                    'summary': doc_info['summary'],
                    'keywords': doc_info['keywords'],
                    'total_chunks': doc_info['total_chunks'],
                    'actual_chunks': doc_info['actual_chunks'],
                    'total_chars': doc_info['total_chars'],
                    'total_tokens': doc_info['total_tokens']
                }
            }
            
        except Exception as e:
            logger.error(f"Document info error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def reindex_document(self, ctx: Context, input: ReindexDocumentInput) -> Dict[str, Any]:
        """
        Force reindexing of a specific document file.
        
        This will reparse the document, regenerate summaries and embeddings,
        and update the index. Useful when a document has been modified or
        if you want to reprocess with updated settings.
        
        Sample usage:
        - Reindex "/home/user/updated-report.pdf" after making changes
        - Force reprocessing of "config/settings.json" with new LLM model
        - Update embeddings for "docs/api-spec.md" after content changes
        - Re-summarize "src/complex-module.py" with improved prompts
        """
        try:
            file_path = Path(input.file_path)
            
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {input.file_path}"
                }
            
            # Parse document
            doc_data = self.parser.parse_file(file_path)
            
            # Process with LLM
            doc_data = await self.processor.process_document(doc_data)
            
            # Index document
            success = await self.indexer.index_document(doc_data)
            
            if success:
                return {
                    "success": True,
                    "message": f"Successfully reindexed: {input.file_path}",
                    "document": {
                        'file_path': str(file_path),
                        'file_name': file_path.name,
                        'summary': doc_data.get('summary', '')[:200],
                        'keywords': doc_data.get('keywords', [])[:5],
                        'num_chunks': doc_data['num_chunks'],
                        'total_chars': doc_data['total_chars']
                    }
                }
            else:
                return {
                    "success": False,
                    "message": f"Document unchanged: {input.file_path}"
                }
                
        except Exception as e:
            logger.error(f"Reindex error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_indexing_stats(self, ctx: Context) -> Dict[str, Any]:
        """
        Get current indexing statistics.
        
        Returns information about the index including document count,
        chunk count, storage size, and file type distribution.
        
        Sample usage:
        - Check how many documents are currently indexed
        - View storage usage and database size
        - See distribution of file types (PDF, markdown, code files, etc.)
        - Monitor indexing progress and collection health
        - Get overview of total chunks and tokens processed
        """
        try:
            stats = await self.indexer.get_stats()
            
            return {
                "success": True,
                "stats": {
                    "total_documents": stats['total_documents'],
                    "total_chunks": stats['total_chunks'],
                    "total_size_mb": round(stats['total_size_bytes'] / (1024 * 1024), 2),
                    "total_chars": stats['total_chars'],
                    "total_tokens": stats['total_tokens'],
                    "file_types": stats['file_types'],
                    "db_path": stats['db_path']
                }
            }
            
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {
                "success": False,
                "error": str(e)
            }