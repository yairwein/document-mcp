"""Local LLM integration for document processing."""

import json
import logging
from typing import Optional, Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import ollama
from ollama import AsyncClient

logger = logging.getLogger(__name__)


class LocalLLM:
    """Interface to local LLM via Ollama."""
    
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url
        self.client = None
        self.async_client = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._available = None
    
    async def initialize(self):
        """Initialize async client."""
        self.async_client = AsyncClient(host=self.base_url)
        self._available = await self.check_availability()
        if self._available:
            logger.info(f"Ollama model {self.model} is available")
        else:
            logger.warning(f"Ollama model {self.model} is not available, using fallback")
    
    async def check_availability(self) -> bool:
        """Check if Ollama and the model are available."""
        try:
            # Check if Ollama is running
            response = await self.async_client.list()
            # Handle both dict and list responses
            if isinstance(response, list):
                model_list = response
            elif isinstance(response, dict):
                model_list = response.get('models', [])
            else:
                model_list = []
            
            # Extract model names safely
            model_names = []
            for m in model_list:
                if isinstance(m, dict) and 'name' in m:
                    model_names.append(m['name'])
                elif isinstance(m, str):
                    model_names.append(m)
            
            # Handle model name variations (e.g., llama3.2:3b vs llama3.2)
            base_model = self.model.split(':')[0]
            available = any(base_model in name for name in model_names)
            
            if not available:
                logger.info(f"Model {self.model} not found. Available models: {model_names}")
                # Try to pull the model
                try:
                    logger.info(f"Attempting to pull model {self.model}...")
                    await self.async_client.pull(self.model)
                    return True
                except Exception as e:
                    logger.error(f"Failed to pull model: {e}")
                    return False
            
            return available
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
            return False
    
    async def summarize_document(self, text: str, max_length: int = 500) -> str:
        """Generate a summary of the document."""
        if not self._available:
            return self._fallback_summary(text, max_length)
        
        try:
            # Truncate very long texts to avoid timeout
            max_input_chars = 8000
            if len(text) > max_input_chars:
                text = text[:max_input_chars] + "..."
            
            prompt = f"""Summarize the following document in {max_length} characters or less. 
Focus on the main topics, key points, and overall purpose of the document.

Document:
{text}

Summary:"""
            
            response = await self.async_client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": max_length
                }
            )
            
            summary = response['response'].strip()
            
            # Ensure summary is not too long
            if len(summary) > max_length:
                summary = summary[:max_length].rsplit(' ', 1)[0] + "..."
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating summary with LLM: {e}")
            return self._fallback_summary(text, max_length)
    
    async def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract keywords from the document."""
        if not self._available:
            return self._fallback_keywords(text, num_keywords)
        
        try:
            # Truncate for keyword extraction
            max_input_chars = 4000
            if len(text) > max_input_chars:
                text = text[:max_input_chars]
            
            prompt = f"""Extract {num_keywords} important keywords or key phrases from this document.
Return only the keywords/phrases as a comma-separated list, nothing else.

Document:
{text}

Keywords:"""
            
            response = await self.async_client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_predict": 200
                }
            )
            
            keywords_text = response['response'].strip()
            keywords = [k.strip() for k in keywords_text.split(',')]
            
            # Clean and validate keywords
            keywords = [k for k in keywords if k and len(k) > 2 and len(k) < 50]
            
            return keywords[:num_keywords]
        
        except Exception as e:
            logger.error(f"Error extracting keywords with LLM: {e}")
            return self._fallback_keywords(text, num_keywords)
    
    async def generate_embedding_text(self, text: str) -> str:
        """Generate text optimized for embedding."""
        if not self._available:
            return text[:1000]  # Simple truncation as fallback
        
        try:
            # For embedding, we want a concise representation
            max_input_chars = 3000
            if len(text) > max_input_chars:
                text = text[:max_input_chars]
            
            prompt = f"""Create a concise version of this text that captures its main meaning and topics.
Keep important keywords and concepts. Maximum 500 characters.

Text:
{text}

Concise version:"""
            
            response = await self.async_client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.2,
                    "num_predict": 500
                }
            )
            
            return response['response'].strip()
        
        except Exception as e:
            logger.error(f"Error generating embedding text: {e}")
            return text[:1000]
    
    def _fallback_summary(self, text: str, max_length: int) -> str:
        """Simple fallback summary when LLM is not available."""
        # Take first few sentences
        sentences = text.split('. ')
        summary = ""
        for sentence in sentences:
            if len(summary) + len(sentence) < max_length:
                summary += sentence + ". "
            else:
                break
        
        if not summary:
            summary = text[:max_length]
        
        return summary.strip()
    
    def _fallback_keywords(self, text: str, num_keywords: int) -> List[str]:
        """Simple keyword extraction when LLM is not available."""
        import re
        from collections import Counter
        
        # Simple word frequency based extraction
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                     'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
                     'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        
        words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Get most common words
        word_counts = Counter(words)
        keywords = [word for word, _ in word_counts.most_common(num_keywords)]
        
        return keywords
    
    async def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)


class DocumentProcessor:
    """Process documents using local LLM."""
    
    def __init__(self, llm: LocalLLM):
        self.llm = llm
    
    async def process_document(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document and add LLM-generated metadata."""
        text = doc_data.get('text', '')
        file_name = doc_data.get('metadata', {}).get('file_name', 'unknown')
        
        if not text:
            logger.info(f"    → Empty document, skipping LLM processing")
            return {
                **doc_data,
                'summary': 'Empty document',
                'keywords': [],
                'embedding_text': ''
            }
        
        logger.info(f"    → Generating summary for {len(text):,} characters...")
        
        # Generate summary, keywords, and embedding text
        summary_task = self.llm.summarize_document(text, max_length=500)
        keywords_task = self.llm.extract_keywords(text, num_keywords=10)
        
        summary, keywords = await asyncio.gather(summary_task, keywords_task)
        
        logger.info(f"    → Generated {len(summary)} char summary and {len(keywords)} keywords: {', '.join(keywords[:3])}{'...' if len(keywords) > 3 else ''}")
        
        # For embedding, use summary + keywords for better representation
        embedding_text = f"{summary} Keywords: {', '.join(keywords)}"
        
        return {
            **doc_data,
            'summary': summary,
            'keywords': keywords,
            'embedding_text': embedding_text
        }
    
    async def process_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple documents in parallel."""
        tasks = [self.process_document(doc) for doc in documents]
        return await asyncio.gather(*tasks)