"""
RAG Chain with Memory
Implements Retrieval Augmented Generation with conversation memory
"""
from typing import List, Dict, Any, Optional
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

from llm_loader import get_manager_llm
from vector_store import get_vector_store


class RAGChatbot:
    """RAG-based chatbot with conversation memory"""
    
    def __init__(
        self,
        memory_key: str = "chat_history",
        return_source_documents: bool = True,
        max_tokens_limit: int = 3000,
        verbose: bool = False
    ):
        """
        Initialize the RAG Chatbot
        
        Args:
            memory_key: Key for storing chat history in memory
            return_source_documents: Whether to return source documents
            max_tokens_limit: Maximum tokens for memory buffer
            verbose: Whether to print verbose output
        """
        self.memory_key = memory_key
        self.return_source_documents = return_source_documents
        self.verbose = verbose
        
        # Initialize LLM
        print("Initializing LLM...")
        self.llm = get_manager_llm()
        
        # Initialize vector store
        print("Initializing vector store...")
        self.vector_store_manager = get_vector_store()
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=True,
            output_key='answer',
            max_token_limit=max_tokens_limit
        )
        
        # Create the conversational chain
        self.chain = None
        self._initialize_chain()
    
    def _get_custom_prompt(self) -> PromptTemplate:
        """
        Create a custom prompt template for the RAG system
        
        Returns:
            PromptTemplate object
        """
        template = """You are a helpful AI assistant that answers questions based on the provided context from documents.
Use the following pieces of context to answer the question at the end. If you don't know the answer or if the context doesn't contain relevant information, just say that you don't know, don't try to make up an answer.

Always provide detailed and accurate answers based on the context. If you reference specific information, try to indicate which part of the context it came from.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Helpful Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )
    
    def _initialize_chain(self):
        """Initialize the conversational retrieval chain"""
        try:
            # Get retriever from vector store
            retriever = self.vector_store_manager.get_retriever(
                search_kwargs={"k": 4}
            )
            
            # Create the conversational retrieval chain
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=self.return_source_documents,
                verbose=self.verbose,
                combine_docs_chain_kwargs={
                    "prompt": self._get_custom_prompt()
                }
            )
            
            print("✓ RAG Chain initialized successfully")
            
        except Exception as e:
            print(f"Warning: Could not initialize chain yet: {str(e)}")
            print("Please add documents to the vector store first.")
            self.chain = None
    
    def reinitialize_chain(self):
        """Reinitialize the chain (useful after adding new documents)"""
        self._initialize_chain()
    
    def chat(self, question: str) -> Dict[str, Any]:
        """
        Send a question to the chatbot and get an answer
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer and source documents
        """
        if self.chain is None:
            self._initialize_chain()
            
            if self.chain is None:
                return {
                    "answer": "I don't have any documents to answer questions from. Please upload documents first.",
                    "source_documents": []
                }
        
        try:
            # Get response from the chain
            response = self.chain.invoke({"question": question})
            
            return {
                "answer": response.get("answer", ""),
                "source_documents": response.get("source_documents", [])
            }
            
        except Exception as e:
            return {
                "answer": f"An error occurred: {str(e)}",
                "source_documents": []
            }
    
    def chat_stream(self, question: str):
        """
        Stream the response from the chatbot
        
        Args:
            question: User's question
            
        Yields:
            Chunks of the response
        """
        if self.chain is None:
            self._initialize_chain()
            
            if self.chain is None:
                yield "I don't have any documents to answer questions from. Please upload documents first."
                return
        
        try:
            # For streaming, we need to use a different approach
            # This is a simplified version - full streaming requires more setup
            response = self.chat(question)
            yield response["answer"]
            
        except Exception as e:
            yield f"An error occurred: {str(e)}"
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Get the chat history
        
        Returns:
            List of message dictionaries
        """
        try:
            messages = self.memory.chat_memory.messages
            history = []
            
            for msg in messages:
                if hasattr(msg, 'type'):
                    history.append({
                        "role": msg.type,
                        "content": msg.content
                    })
            
            return history
        except:
            return []
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()
        print("✓ Conversation memory cleared")
    
    def add_documents_to_store(self, documents: List):
        """
        Add documents to the vector store
        
        Args:
            documents: List of Document objects
        """
        self.vector_store_manager.add_documents(documents)
        self.reinitialize_chain()
        print("✓ Documents added and chain reinitialized")
    
    def get_relevant_documents(self, query: str, k: int = 4):
        """
        Get relevant documents without generating an answer
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of relevant documents with scores
        """
        return self.vector_store_manager.similarity_search_with_score(
            query=query,
            k=k
        )


# Global chatbot instance
_chatbot: Optional[RAGChatbot] = None


def get_chatbot() -> RAGChatbot:
    """
    Get the global chatbot instance (singleton pattern)
    
    Returns:
        RAGChatbot instance
    """
    global _chatbot
    if _chatbot is None:
        _chatbot = RAGChatbot()
    return _chatbot
