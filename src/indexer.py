"""LanceDB indexing operations for document storage and retrieval."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import lancedb
from sentence_transformers import SentenceTransformer
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Manage document indexing with LanceDB."""
    
    def __init__(self, db_path: Path, embedding_model: str):
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.db = None
        self.catalog_table = None
        self.chunks_table = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """Initialize the database and embedding model."""
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize LanceDB
        logger.info(f"Initializing LanceDB at: {self.db_path}")
        self.db = await asyncio.get_event_loop().run_in_executor(
            self.executor, lancedb.connect, str(self.db_path)
        )
        
        # Create or open tables
        await self._ensure_tables()
        
    async def _ensure_tables(self):
        """Ensure catalog and chunks tables exist."""
        # Get embedding dimension
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Catalog table schema - using object for embedding column
        catalog_schema = {
            "file_path": str,
            "file_name": str,
            "file_hash": str,
            "file_size": int,
            "file_type": str,
            "modified_time": str,
            "indexed_time": str,
            "summary": str,
            "keywords": str,  # JSON array as string
            "total_chunks": int,
            "total_chars": int,
            "total_tokens": int,
            "embedding": object  # Vector for document-level search
        }
        
        # Chunks table schema - using object for embedding column
        chunks_schema = {
            "file_path": str,
            "file_hash": str,
            "chunk_id": int,
            "chunk_text": str,
            "start_pos": int,
            "end_pos": int,
            "char_count": int,
            "token_count": int,
            "embedding": object  # Vector for chunk-level search
        }
        
        # Check if tables exist
        existing_tables = await asyncio.get_event_loop().run_in_executor(
            self.executor, self.db.table_names
        )
        
        if "catalog" not in existing_tables:
            # Create catalog table with initial data
            logger.info("Creating catalog table")
            # Create a single row with dummy data to establish schema
            dummy_embedding = np.zeros(embedding_dim, dtype=np.float32)
            initial_data = pd.DataFrame({
                "file_path": ["dummy"],
                "file_name": ["dummy"],
                "file_hash": ["dummy"],
                "file_size": [0],
                "file_type": ["dummy"],
                "modified_time": ["2024-01-01"],
                "indexed_time": ["2024-01-01"],
                "summary": ["dummy"],
                "keywords": ["[]"],
                "total_chunks": [0],
                "total_chars": [0],
                "total_tokens": [0],
                "embedding": [dummy_embedding]
            })
            self.catalog_table = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.db.create_table, "catalog", initial_data
            )
            # Remove the dummy row
            await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: self.catalog_table.delete("file_path = 'dummy'")
            )
        else:
            self.catalog_table = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.db.open_table, "catalog"
            )
        
        if "chunks" not in existing_tables:
            # Create chunks table with initial data
            logger.info("Creating chunks table")
            dummy_embedding = np.zeros(embedding_dim, dtype=np.float32)
            initial_data = pd.DataFrame({
                "file_path": ["dummy"],
                "file_hash": ["dummy"],
                "chunk_id": [0],
                "chunk_text": ["dummy"],
                "start_pos": [0],
                "end_pos": [0],
                "char_count": [0],
                "token_count": [0],
                "embedding": [dummy_embedding]
            })
            self.chunks_table = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.db.create_table, "chunks", initial_data
            )
            # Remove the dummy row
            await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: self.chunks_table.delete("file_path = 'dummy'")
            )
        else:
            self.chunks_table = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.db.open_table, "chunks"
            )
    
    async def index_document(self, doc_data: Dict[str, Any]) -> bool:
        """Index a document and its chunks."""
        try:
            metadata = doc_data['metadata']
            file_path = metadata['file_path']
            file_hash = metadata['file_hash']
            file_name = metadata['file_name']
            num_chunks = doc_data['num_chunks']
            
            logger.info(f"    → Checking if document is already indexed...")
            # Check if document already indexed with same hash
            if await self.is_document_indexed(file_path, file_hash):
                logger.info(f"    → Document already indexed with same hash: {file_name}")
                return False
            
            logger.info(f"    → Removing old version if exists...")
            # Remove old version if exists
            await self.remove_document(file_path)
            
            logger.info(f"    → Generating document-level embedding...")
            # Generate embeddings for summary/keywords
            embedding_text = doc_data.get('embedding_text', doc_data.get('summary', ''))
            if embedding_text:
                doc_embedding = await self._generate_embedding(embedding_text)
            else:
                # Use first chunk as fallback
                first_chunk_text = doc_data['chunks'][0]['text'] if doc_data['chunks'] else ""
                doc_embedding = await self._generate_embedding(first_chunk_text)
            
            logger.info(f"    → Generated document embedding ({len(doc_embedding)} dimensions)")
            
            logger.info(f"    → Adding document to catalog...")
            # Prepare catalog entry
            catalog_entry = {
                "file_path": file_path,
                "file_name": metadata['file_name'],
                "file_hash": file_hash,
                "file_size": metadata['file_size'],
                "file_type": metadata['file_type'],
                "modified_time": metadata['modified_time'],
                "indexed_time": datetime.now().isoformat(),
                "summary": doc_data.get('summary', ''),
                "keywords": json.dumps(doc_data.get('keywords', [])),
                "total_chunks": doc_data['num_chunks'],
                "total_chars": doc_data['total_chars'],
                "total_tokens": doc_data['total_tokens'],
                "embedding": doc_embedding.astype(np.float32)
            }
            
            # Add to catalog
            catalog_df = pd.DataFrame([catalog_entry])
            await asyncio.get_event_loop().run_in_executor(
                self.executor, self.catalog_table.add, catalog_df
            )
            logger.info(f"    → Document added to catalog")
            
            # Index chunks
            if doc_data['chunks']:
                logger.info(f"    → Processing {num_chunks} chunks for embedding...")
                chunk_entries = []
                
                # Process chunks with progress logging
                for i, chunk in enumerate(doc_data['chunks']):
                    if (i + 1) % 5 == 0 or i == len(doc_data['chunks']) - 1:
                        logger.info(f"    → Generating embeddings: {i + 1}/{num_chunks} chunks")
                    
                    # Generate embedding for chunk
                    chunk_embedding = await self._generate_embedding(chunk['text'])
                    
                    chunk_entry = {
                        "file_path": file_path,
                        "file_hash": file_hash,
                        "chunk_id": chunk['chunk_id'],
                        "chunk_text": chunk['text'],
                        "start_pos": chunk['start_pos'],
                        "end_pos": chunk['end_pos'],
                        "char_count": chunk['char_count'],
                        "token_count": chunk['token_count'],
                        "embedding": chunk_embedding.astype(np.float32)
                    }
                    chunk_entries.append(chunk_entry)
                
                logger.info(f"    → Storing {num_chunks} chunks to database...")
                # Add chunks in batch
                chunks_df = pd.DataFrame(chunk_entries)
                await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.chunks_table.add, chunks_df
                )
                logger.info(f"    → All {num_chunks} chunks stored successfully")
            else:
                logger.warning(f"    → No chunks to index for {file_name}")
            
            logger.info(f"Successfully indexed: {file_name} ({doc_data['num_chunks']} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing document {doc_data.get('metadata', {}).get('file_path', 'unknown')}: {e}")
            raise
    
    async def is_document_indexed(self, file_path: str, file_hash: str) -> bool:
        """Check if document is already indexed with the same hash."""
        try:
            query = f"file_path = '{file_path}' AND file_hash = '{file_hash}'"
            results = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.catalog_table.search().where(query).limit(1).to_pandas()
            )
            return len(results) > 0
        except Exception:
            return False
    
    async def remove_document(self, file_path: str):
        """Remove a document and its chunks from the index."""
        try:
            # Check if document exists first
            info = await self.get_document_info(file_path)
            if not info:
                return False
            
            # Remove from catalog
            query = f"file_path = '{file_path}'"
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.catalog_table.delete(query)
            )
            
            # Remove chunks
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.chunks_table.delete(query)
            )
            
            logger.info(f"Removed document from index: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error removing document {file_path}: {e}")
            return False
    
    async def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for documents using semantic search."""
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        query_embedding = query_embedding.astype(np.float32)
        
        # Search in catalog - specify vector column name
        results = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.catalog_table.search(
                query_embedding,
                vector_column_name="embedding"
            )
            .limit(limit)
            .to_pandas()
        )
        
        # Convert to list of dicts
        documents = []
        for _, row in results.iterrows():
            doc = row.to_dict()
            doc['keywords'] = json.loads(doc.get('keywords', '[]'))
            doc.pop('embedding', None)  # Remove embedding from response
            documents.append(doc)
        
        return documents
    
    async def search_chunks(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for specific chunks using semantic search."""
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        query_embedding = query_embedding.astype(np.float32)
        
        # Search in chunks - specify vector column name
        results = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.chunks_table.search(
                query_embedding,
                vector_column_name="embedding"
            )
            .limit(limit)
            .to_pandas()
        )
        
        # Convert to list of dicts
        chunks = []
        for _, row in results.iterrows():
            chunk = row.to_dict()
            chunk.pop('embedding', None)  # Remove embedding from response
            chunks.append(chunk)
        
        return chunks
    
    async def get_catalog(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all documents in the catalog."""
        results = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.catalog_table.to_pandas()
        )
        
        # Sort by indexed time (newest first)
        results = results.sort_values('indexed_time', ascending=False)
        
        # Apply pagination
        results = results.iloc[skip:skip + limit]
        
        # Convert to list of dicts
        documents = []
        for _, row in results.iterrows():
            doc = row.to_dict()
            doc['keywords'] = json.loads(doc.get('keywords', '[]'))
            doc.pop('embedding', None)  # Remove embedding from response
            documents.append(doc)
        
        return documents
    
    async def get_document_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific document."""
        query = f"file_path = '{file_path}'"
        results = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.catalog_table.search().where(query).limit(1).to_pandas()
        )
        
        if len(results) == 0:
            return None
        
        doc = results.iloc[0].to_dict()
        doc['keywords'] = json.loads(doc.get('keywords', '[]'))
        doc.pop('embedding', None)
        
        # Get chunk count
        chunk_results = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.chunks_table.search().where(query).to_pandas()
        )
        doc['actual_chunks'] = len(chunk_results)
        
        return doc
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        catalog_df = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.catalog_table.to_pandas()
        )
        
        chunks_df = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.chunks_table.to_pandas()
        )
        
        return {
            "total_documents": len(catalog_df),
            "total_chunks": len(chunks_df),
            "total_size_bytes": int(catalog_df['file_size'].sum()) if len(catalog_df) > 0 else 0,
            "total_chars": int(catalog_df['total_chars'].sum()) if len(catalog_df) > 0 else 0,
            "total_tokens": int(catalog_df['total_tokens'].sum()) if len(catalog_df) > 0 else 0,
            "file_types": catalog_df['file_type'].value_counts().to_dict() if len(catalog_df) > 0 else {},
            "db_path": str(self.db_path)
        }
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if not text:
            # Return zero vector for empty text
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension())
        
        # Run in executor to avoid blocking
        embedding = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.embedding_model.encode,
            text
        )
        
        return embedding
    
    async def _generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts in batch."""
        if not texts:
            return []
        
        # Run in executor to avoid blocking
        embeddings = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.embedding_model.encode,
            texts
        )
        
        # Convert to list of individual embeddings
        return [embedding for embedding in embeddings]
    
    async def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)