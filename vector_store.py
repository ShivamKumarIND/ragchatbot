"""
Vector Store Manager
Manages document embeddings and similarity search using ChromaDB
"""
import os
from typing import List, Optional
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


class VectorStoreManager:
    """Manages vector store for document embeddings"""
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "rag_documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the Vector Store Manager
        
        Args:
            persist_directory: Directory to persist the vector store
            collection_name: Name of the collection
            embedding_model: HuggingFace embedding model name
        """
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIRECTORY",
            "./chroma_db"
        )
        self.collection_name = collection_name
        
        # Initialize embeddings
        print(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize or load vector store
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize or load existing vector store"""
        if os.path.exists(self.persist_directory):
            print(f"Loading existing vector store from {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
        else:
            print(f"Creating new vector store at {self.persist_directory}")
            # Will be created when first documents are added
            self.vectorstore = None
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs
        """
        if not documents:
            print("No documents to add")
            return []
        
        if self.vectorstore is None:
            # Create new vector store with documents
            print(f"Creating vector store with {len(documents)} documents")
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
        else:
            # Add to existing vector store
            print(f"Adding {len(documents)} documents to existing vector store")
            ids = self.vectorstore.add_documents(documents)
            return ids
        
        # Persist the changes
        if hasattr(self.vectorstore, 'persist'):
            self.vectorstore.persist()
        
        print(f"✓ Successfully added {len(documents)} documents to vector store")
        return []
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query: Query text
            k: Number of documents to return
            filter: Metadata filter dictionary
            
        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            print("Vector store is empty. Please add documents first.")
            return []
        
        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[tuple]:
        """
        Search for similar documents with relevance scores
        
        Args:
            query: Query text
            k: Number of documents to return
            filter: Metadata filter dictionary
            
        Returns:
            List of tuples (document, score)
        """
        if self.vectorstore is None:
            print("Vector store is empty. Please add documents first.")
            return []
        
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get a retriever interface for the vector store
        
        Args:
            search_kwargs: Arguments for retriever (e.g., {'k': 4})
            
        Returns:
            Retriever object
        """
        if self.vectorstore is None:
            raise ValueError("Vector store is empty. Please add documents first.")
        
        search_kwargs = search_kwargs or {"k": 4}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def delete_collection(self):
        """Delete the entire collection"""
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            self.vectorstore = None
            print(f"✓ Deleted collection: {self.collection_name}")
    
    def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection
        
        Returns:
            Number of documents
        """
        if self.vectorstore is None:
            return 0
        
        try:
            collection = self.vectorstore._collection
            return collection.count()
        except:
            return 0
    
    def list_collections(self) -> List[str]:
        """
        List all available collections
        
        Returns:
            List of collection names
        """
        if self.vectorstore is None:
            return []
        
        try:
            client = self.vectorstore._client
            collections = client.list_collections()
            return [col.name for col in collections]
        except:
            return []


# Global instance for easy access
_vector_store: Optional[VectorStoreManager] = None


def get_vector_store() -> VectorStoreManager:
    """
    Get the global vector store instance (singleton pattern)
    
    Returns:
        VectorStoreManager instance
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreManager()
    return _vector_store
