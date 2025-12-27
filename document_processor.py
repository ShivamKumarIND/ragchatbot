"""
Document Processor
Handles loading and processing of various document types
Supports: PDF, DOCX, HTML, CSV, XLSX, TXT
"""
import os
from typing import List, Optional
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    TextLoader
)
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()


class DocumentProcessor:
    """Process and split documents for RAG"""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize the Document Processor
        
        Args:
            chunk_size: Size of text chunks (default from env or 1000)
            chunk_overlap: Overlap between chunks (default from env or 200)
        """
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "200"))
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Supported file extensions and their loaders
        self.loader_mapping = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".doc": Docx2txtLoader,
            ".html": UnstructuredHTMLLoader,
            ".htm": UnstructuredHTMLLoader,
            ".csv": CSVLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".xls": UnstructuredExcelLoader,
            ".txt": TextLoader,
        }
    
    def get_loader_for_file(self, file_path: str):
        """
        Get the appropriate document loader for a file
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Appropriate document loader instance
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in self.loader_mapping:
            raise ValueError(
                f"Unsupported file type: {file_extension}. "
                f"Supported types: {', '.join(self.loader_mapping.keys())}"
            )
        
        loader_class = self.loader_mapping[file_extension]
        return loader_class(file_path)
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document from file
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            loader = self.get_loader_for_file(file_path)
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata["source"] = os.path.basename(file_path)
                doc.metadata["file_path"] = file_path
            
            return documents
        
        except Exception as e:
            raise Exception(f"Error loading document {file_path}: {str(e)}")
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of split Document objects
        """
        return self.text_splitter.split_documents(documents)
    
    def process_document(self, file_path: str) -> List[Document]:
        """
        Load and split a document in one step
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of split Document objects ready for embedding
        """
        documents = self.load_document(file_path)
        split_docs = self.split_documents(documents)
        
        print(f"Processed {file_path}:")
        print(f"  - Loaded {len(documents)} document(s)")
        print(f"  - Split into {len(split_docs)} chunks")
        
        return split_docs
    
    def process_multiple_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Process multiple documents at once
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Combined list of split documents
        """
        all_splits = []
        
        for file_path in file_paths:
            try:
                splits = self.process_document(file_path)
                all_splits.extend(splits)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        print(f"\nTotal processed: {len(all_splits)} chunks from {len(file_paths)} files")
        return all_splits
    
    def process_directory(
        self,
        directory_path: str,
        recursive: bool = False
    ) -> List[Document]:
        """
        Process all supported documents in a directory
        
        Args:
            directory_path: Path to directory containing documents
            recursive: Whether to search subdirectories
            
        Returns:
            List of split documents from all files in directory
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Not a directory: {directory_path}")
        
        file_paths = []
        
        if recursive:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if Path(file_path).suffix.lower() in self.loader_mapping:
                        file_paths.append(file_path)
        else:
            for file in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path) and Path(file_path).suffix.lower() in self.loader_mapping:
                    file_paths.append(file_path)
        
        print(f"Found {len(file_paths)} supported documents in {directory_path}")
        return self.process_multiple_documents(file_paths)
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions
        
        Returns:
            List of supported file extensions
        """
        return list(self.loader_mapping.keys())
