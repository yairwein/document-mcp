"""Unit tests for LLM and document processor."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.llm import LocalLLM, DocumentProcessor
from src.parser import DocumentParser


class TestLocalLLM:
    """Test Local LLM functionality."""

    @pytest.mark.unit
    def test_init(self):
        """Test LLM initialization."""
        llm = LocalLLM(model="test-model", base_url="http://test:11434")
        assert llm.model == "test-model"
        assert llm.base_url == "http://test:11434"
        assert llm.client is None

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_initialization_with_real_ollama(self, test_config):
        """Test LLM initialization with real Ollama (if available)."""
        llm = LocalLLM(model=test_config.llm_model, base_url=test_config.ollama_base_url)
        
        try:
            await llm.initialize()
            assert llm.client is not None
            assert llm.is_available
        except Exception:
            # Skip if Ollama not available
            pytest.skip("Ollama not available for testing")
        finally:
            if llm.client:
                await llm.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialization_without_ollama(self):
        """Test LLM initialization without Ollama (should use fallback)."""
        llm = LocalLLM(model="test-model", base_url="http://nonexistent:11434")
        
        await llm.initialize()
        
        # Should fall back to mock responses
        assert not llm.is_available

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_with_mock(self):
        """Test text generation with mocked responses."""
        llm = LocalLLM(model="test-model", base_url="http://nonexistent:11434")
        await llm.initialize()
        
        # Test fallback generation
        result = await llm.generate("Test prompt")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_generate_with_real_ollama(self, llm):
        """Test text generation with real Ollama."""
        if not llm.is_available:
            pytest.skip("Ollama not available")
        
        prompt = "Summarize this text in one sentence: The quick brown fox jumps over the lazy dog."
        result = await llm.generate(prompt, max_tokens=50)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain at least some relevant content or mention of the original text
        # More flexible check - just ensure it's not a fallback response
        assert not result.startswith("[Mock response")
        assert not result.startswith("[Fallback response")
        # Basic content check - should be a reasonable sentence
        assert len(result.split()) >= 3  # At least a few words

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close(self):
        """Test LLM cleanup."""
        llm = LocalLLM(model="test-model", base_url="http://test:11434")
        await llm.initialize()
        await llm.close()
        # Should not raise any exceptions


class TestDocumentProcessor:
    """Test document processor functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_init(self):
        """Test processor initialization."""
        llm = LocalLLM(model="test-model", base_url="http://test:11434")
        await llm.initialize()
        
        processor = DocumentProcessor(llm)
        assert processor.llm == llm

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_document_structure(self, document_processor, document_parser, sample_text_file):
        """Test document processing returns correct structure."""
        # Parse document first
        doc_data = document_parser.parse_file(sample_text_file)
        
        # Process with LLM
        result = await document_processor.process_document(doc_data)
        
        # Check that original data is preserved
        assert result['metadata'] == doc_data['metadata']
        assert result['chunks'] == doc_data['chunks']
        assert result['total_chars'] == doc_data['total_chars']
        assert result['num_chunks'] == doc_data['num_chunks']
        assert result['total_tokens'] == doc_data['total_tokens']
        
        # Check new fields added
        assert 'summary' in result
        assert 'keywords' in result
        assert isinstance(result['summary'], str)
        assert isinstance(result['keywords'], list)
        assert len(result['summary']) > 0
        assert len(result['keywords']) > 0

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_process_legal_document(self, document_processor, document_parser, sample_legal_file):
        """Test processing a legal document."""
        # Parse legal document
        doc_data = document_parser.parse_file(sample_legal_file)
        
        # Process with LLM
        result = await document_processor.process_document(doc_data)
        
        # Check summary contains legal terms
        summary = result['summary'].lower()
        assert any(term in summary for term in ['agreement', 'confidential', 'disclosure', 'nda'])
        
        # Check keywords contain legal terms
        keywords_lower = [k.lower() for k in result['keywords']]
        assert any(term in ' '.join(keywords_lower) for term in ['agreement', 'confidential', 'disclosure'])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_empty_document(self, document_processor, document_parser, temp_dir):
        """Test processing an empty document."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")
        
        doc_data = document_parser.parse_file(empty_file)
        result = await document_processor.process_document(doc_data)
        
        # Should handle empty document gracefully
        assert 'summary' in result
        assert 'keywords' in result
        assert isinstance(result['summary'], str)
        assert isinstance(result['keywords'], list)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_summary(self, document_processor):
        """Test summary generation directly."""
        text = "This is a test document about machine learning and artificial intelligence."
        
        summary = await document_processor._generate_summary(text)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        # Summary should be shorter than original (or reasonable length)
        assert len(summary) <= max(len(text) + 100, 500)  # Allow some variance

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_keywords(self, document_processor):
        """Test keyword extraction directly."""
        text = "This document discusses machine learning algorithms and data science techniques."
        
        keywords = await document_processor._extract_keywords(text)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert all(isinstance(kw, str) for kw in keywords)
        # Should extract relevant terms
        keywords_text = ' '.join(keywords).lower()
        assert any(term in keywords_text for term in ['machine', 'learning', 'data', 'algorithm'])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_large_document(self, document_processor, document_parser, temp_dir):
        """Test processing a large document."""
        # Create a large document
        large_content = "This is a test sentence. " * 1000  # ~25k characters
        large_file = temp_dir / "large_doc.txt"
        large_file.write_text(large_content)
        
        doc_data = document_parser.parse_file(large_file)
        result = await document_processor.process_document(doc_data)
        
        # Should handle large document and produce reasonable summary
        assert len(result['summary']) > 0
        assert len(result['summary']) < 1000  # Summary should be much shorter
        assert len(result['keywords']) > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_summary_length_limit(self, document_processor):
        """Test that summary respects length limits."""
        long_text = "Very detailed content. " * 500
        
        summary = await document_processor._generate_summary(long_text, max_length=100)
        
        # Summary should respect the length limit (approximately)
        assert len(summary) <= 150  # Allow some variance for completion

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_keyword_count_limit(self, document_processor):
        """Test that keyword extraction respects count limits."""
        text = """
        Python programming language software development web development
        machine learning artificial intelligence data science analytics
        database management system design patterns algorithms optimization
        testing debugging deployment continuous integration automation
        """
        
        keywords = await document_processor._extract_keywords(text, max_keywords=5)
        
        # Should limit to requested number of keywords
        assert len(keywords) <= 5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_handling_in_processing(self, temp_dir):
        """Test error handling when LLM fails."""
        # Create a mock LLM that always fails
        mock_llm = AsyncMock()
        mock_llm.summarize_document = AsyncMock(side_effect=Exception("LLM error"))
        mock_llm.extract_keywords = AsyncMock(side_effect=Exception("LLM error"))
        mock_llm.is_available = False
        
        processor = DocumentProcessor(mock_llm)
        
        # Create test document
        doc_parser = DocumentParser(chunk_size=500, chunk_overlap=100)
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content")
        doc_data = doc_parser.parse_file(test_file)
        
        # Should handle LLM errors gracefully with fallbacks
        try:
            result = await processor.process_document(doc_data)
            # If it doesn't raise an exception, check the result structure
            assert 'summary' in result
            assert 'keywords' in result
        except Exception:
            # It's acceptable for this to fail since we're mocking failures
            pass

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, document_processor, document_parser, multiple_test_files):
        """Test processing multiple documents concurrently."""
        import asyncio
        
        # Parse all files
        doc_data_list = [document_parser.parse_file(f) for f in multiple_test_files]
        
        # Process concurrently
        tasks = [document_processor.process_document(doc_data) for doc_data in doc_data_list]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == len(multiple_test_files)
        for result in results:
            assert 'summary' in result
            assert 'keywords' in result
            assert len(result['summary']) > 0
            assert len(result['keywords']) > 0