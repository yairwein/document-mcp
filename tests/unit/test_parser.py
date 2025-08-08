"""Unit tests for document parser."""

import pytest
from pathlib import Path
from src.parser import DocumentParser, is_supported_file


class TestDocumentParser:
    """Test document parser functionality."""

    @pytest.mark.unit
    def test_init(self):
        """Test parser initialization."""
        parser = DocumentParser(chunk_size=1000, chunk_overlap=200)
        assert parser.chunk_size == 1000
        assert parser.chunk_overlap == 200
        assert parser.tokenizer is not None

    @pytest.mark.unit
    def test_parse_text_file(self, document_parser, sample_text_file):
        """Test parsing a text file."""
        doc_data = document_parser.parse_file(sample_text_file)
        
        # Check metadata
        assert doc_data['metadata']['file_name'] == sample_text_file.name
        assert doc_data['metadata']['file_path'] == str(sample_text_file)
        assert doc_data['metadata']['file_type'] == '.txt'
        assert doc_data['metadata']['file_size'] > 0
        
        # Check content
        assert doc_data['total_chars'] > 0
        assert doc_data['num_chunks'] > 0
        assert len(doc_data['chunks']) == doc_data['num_chunks']
        assert doc_data['total_tokens'] > 0
        
        # Check chunks structure
        for chunk in doc_data['chunks']:
            assert 'text' in chunk
            assert 'start_pos' in chunk
            assert 'end_pos' in chunk
            assert 'char_count' in chunk
            assert 'token_count' in chunk
            assert chunk['char_count'] > 0
            assert chunk['token_count'] > 0

    @pytest.mark.unit
    def test_chunk_overlap(self, temp_dir):
        """Test chunk overlap functionality."""
        parser = DocumentParser(chunk_size=100, chunk_overlap=20)
        
        # Create a longer text file
        content = "This is a test. " * 50  # ~800 characters
        test_file = temp_dir / "overlap_test.txt"
        test_file.write_text(content)
        
        doc_data = parser.parse_file(test_file)
        
        # Should have multiple chunks with overlap > chunk_size
        assert doc_data['num_chunks'] > 1
        
        # Check overlap between consecutive chunks
        chunks = doc_data['chunks']
        if len(chunks) > 1:
            # There should be some overlap in positions
            assert chunks[1]['start_pos'] < chunks[0]['end_pos']

    @pytest.mark.unit
    def test_file_hash_generation(self, document_parser, sample_text_file):
        """Test file hash generation."""
        doc_data = document_parser.parse_file(sample_text_file)
        file_hash = doc_data['metadata']['file_hash']
        
        assert file_hash is not None
        assert len(file_hash) == 64  # SHA256 hex string length
        assert all(c in '0123456789abcdef' for c in file_hash)

    @pytest.mark.unit
    def test_same_file_same_hash(self, document_parser, temp_dir):
        """Test that identical files produce identical hashes."""
        content = "Test content for hash comparison"
        
        file1 = temp_dir / "test1.txt"
        file2 = temp_dir / "test2.txt"
        
        file1.write_text(content)
        file2.write_text(content)
        
        doc1 = document_parser.parse_file(file1)
        doc2 = document_parser.parse_file(file2)
        
        assert doc1['metadata']['file_hash'] == doc2['metadata']['file_hash']

    @pytest.mark.unit
    def test_different_files_different_hash(self, document_parser, temp_dir):
        """Test that different files produce different hashes."""
        file1 = temp_dir / "test1.txt"
        file2 = temp_dir / "test2.txt"
        
        file1.write_text("Content 1")
        file2.write_text("Content 2")
        
        doc1 = document_parser.parse_file(file1)
        doc2 = document_parser.parse_file(file2)
        
        assert doc1['metadata']['file_hash'] != doc2['metadata']['file_hash']

    @pytest.mark.unit
    def test_empty_file(self, document_parser, temp_dir):
        """Test parsing an empty file."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")
        
        doc_data = document_parser.parse_file(empty_file)
        
        assert doc_data['total_chars'] == 0
        assert doc_data['num_chunks'] == 0
        assert len(doc_data['chunks']) == 0
        assert doc_data['total_tokens'] == 0

    @pytest.mark.unit
    def test_nonexistent_file(self, document_parser, temp_dir):
        """Test parsing a nonexistent file raises appropriate error."""
        nonexistent = temp_dir / "does_not_exist.txt"
        
        with pytest.raises(FileNotFoundError):
            document_parser.parse_file(nonexistent)

    @pytest.mark.unit
    def test_is_supported_file(self):
        """Test file type support checking."""
        extensions = ['.pdf', '.docx', '.txt', '.md']
        
        assert is_supported_file(Path('test.pdf'), extensions)
        assert is_supported_file(Path('test.docx'), extensions)
        assert is_supported_file(Path('test.txt'), extensions)
        assert is_supported_file(Path('test.md'), extensions)
        
        assert not is_supported_file(Path('test.jpg'), extensions)
        assert not is_supported_file(Path('test.exe'), extensions)
        assert not is_supported_file(Path('test'), extensions)

    @pytest.mark.unit
    def test_case_insensitive_extensions(self):
        """Test that file extension checking is case insensitive."""
        extensions = ['.pdf', '.txt']
        
        assert is_supported_file(Path('test.PDF'), extensions)
        assert is_supported_file(Path('test.TXT'), extensions)
        assert is_supported_file(Path('test.Pdf'), extensions)
        assert is_supported_file(Path('test.TxT'), extensions)

    @pytest.mark.unit
    def test_token_counting(self, temp_dir):
        """Test token counting accuracy."""
        # Use much larger chunk size to create single chunk for small content
        parser = DocumentParser(chunk_size=5000, chunk_overlap=50)
        
        # Use predictable content for token counting
        content = "word " * 100  # 100 words, ~500 chars
        test_file = temp_dir / "token_test.txt"
        test_file.write_text(content)
        
        doc_data = parser.parse_file(test_file)
        
        # Should have reasonable token count (not exact due to tokenization differences)
        assert doc_data['total_tokens'] > 90  # Allow some variance
        assert doc_data['total_tokens'] < 120
        
        # With large chunk size, should create single chunk for small content
        assert doc_data['num_chunks'] >= 1
        
        # Test that individual chunks have token counts
        for chunk in doc_data['chunks']:
            assert chunk['token_count'] > 0

    @pytest.mark.unit
    def test_metadata_completeness(self, document_parser, sample_text_file):
        """Test that all required metadata is present."""
        doc_data = document_parser.parse_file(sample_text_file)
        
        required_metadata = [
            'file_name', 'file_path', 'file_type', 'file_size', 
            'file_hash', 'modified_time'
        ]
        
        for field in required_metadata:
            assert field in doc_data['metadata']
            assert doc_data['metadata'][field] is not None

    @pytest.mark.unit
    def test_large_chunk_size(self, temp_dir):
        """Test parser with very large chunk size."""
        parser = DocumentParser(chunk_size=10000, chunk_overlap=100)
        
        content = "This is test content. " * 100
        test_file = temp_dir / "large_chunk_test.txt"
        test_file.write_text(content)
        
        doc_data = parser.parse_file(test_file)
        
        # Should create fewer chunks with large chunk size
        assert doc_data['num_chunks'] >= 1
        if doc_data['num_chunks'] == 1:
            # Single chunk should contain all content
            assert doc_data['chunks'][0]['char_count'] == doc_data['total_chars']

    @pytest.mark.unit
    def test_zero_overlap(self, temp_dir):
        """Test parser with zero overlap."""
        parser = DocumentParser(chunk_size=100, chunk_overlap=0)
        
        content = "This is a test. " * 20
        test_file = temp_dir / "no_overlap_test.txt"
        test_file.write_text(content)
        
        doc_data = parser.parse_file(test_file)
        
        # Chunks should not overlap
        chunks = doc_data['chunks']
        for i in range(len(chunks) - 1):
            assert chunks[i]['end_pos'] <= chunks[i + 1]['start_pos']