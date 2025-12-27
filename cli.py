"""
Command Line Interface for RAG Chatbot
Simple CLI to interact with the RAG system
"""
import sys
import os
from pathlib import Path

from document_processor import DocumentProcessor
from vector_store import get_vector_store
from rag_chain import get_chatbot
from llm_loader import get_llm_loader


def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 60)
    print("  RAG CHATBOT - Document-Based Q&A System")
    print("=" * 60)
    print()


def print_menu():
    """Print main menu"""
    print("\nAvailable Commands:")
    print("  1. upload   - Upload and process documents")
    print("  2. chat     - Start chatting with the assistant")
    print("  3. search   - Search for relevant documents")
    print("  4. history  - View chat history")
    print("  5. clear    - Clear conversation memory")
    print("  6. status   - View system status")
    print("  7. help     - Show this menu")
    print("  8. exit     - Exit the application")
    print()


def upload_documents():
    """Upload and process documents"""
    print("\n--- Document Upload ---")
    
    path = input("Enter file path or directory path: ").strip()
    
    if not os.path.exists(path):
        print("❌ Path does not exist!")
        return
    
    try:
        processor = DocumentProcessor()
        
        if os.path.isfile(path):
            # Process single file
            split_docs = processor.process_document(path)
        else:
            # Process directory
            split_docs = processor.process_directory(path)
        
        # Add to vector store
        vector_store = get_vector_store()
        vector_store.add_documents(split_docs)
        
        # Reinitialize chatbot
        chatbot = get_chatbot()
        chatbot.reinitialize_chain()
        
        print(f"✓ Successfully processed and indexed {len(split_docs)} document chunks")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def chat_mode():
    """Interactive chat mode"""
    print("\n--- Chat Mode ---")
    print("Type 'back' to return to main menu\n")
    
    chatbot = get_chatbot()
    
    while True:
        question = input("\nYou: ").strip()
        
        if not question:
            continue
        
        if question.lower() in ['back', 'exit', 'quit']:
            break
        
        try:
            response = chatbot.chat(question)
            
            print(f"\nAssistant: {response['answer']}")
            
            if response.get('source_documents'):
                print(f"\n📚 Sources ({len(response['source_documents'])} documents):")
                for i, doc in enumerate(response['source_documents'][:2], 1):
                    source = doc.metadata.get('source', 'Unknown')
                    print(f"  {i}. {source}")
        
        except Exception as e:
            print(f"❌ Error: {str(e)}")


def search_documents():
    """Search for relevant documents"""
    print("\n--- Document Search ---")
    
    query = input("Enter search query: ").strip()
    
    if not query:
        return
    
    try:
        k = input("Number of results (default 4): ").strip()
        k = int(k) if k else 4
        
        chatbot = get_chatbot()
        results = chatbot.get_relevant_documents(query, k)
        
        print(f"\n📚 Found {len(results)} relevant documents:\n")
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. Relevance Score: {score:.4f}")
            print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"   Content: {doc.page_content[:150]}...")
            print()
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def view_history():
    """View chat history"""
    print("\n--- Chat History ---")
    
    try:
        chatbot = get_chatbot()
        history = chatbot.get_chat_history()
        
        if not history:
            print("No chat history available.")
            return
        
        for i, msg in enumerate(history, 1):
            role = msg['role'].upper()
            content = msg['content']
            print(f"\n{i}. {role}:")
            print(f"   {content[:200]}{'...' if len(content) > 200 else ''}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def clear_memory():
    """Clear conversation memory"""
    confirm = input("Are you sure you want to clear chat history? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        try:
            chatbot = get_chatbot()
            chatbot.clear_memory()
            print("✓ Chat history cleared")
        except Exception as e:
            print(f"❌ Error: {str(e)}")


def show_status():
    """Show system status"""
    print("\n--- System Status ---")
    
    try:
        # Vector store info
        vector_store = get_vector_store()
        doc_count = vector_store.get_collection_count()
        print(f"📚 Documents Indexed: {doc_count}")
        
        # LLM info
        llm_loader = get_llm_loader()
        print(f"🤖 Current LLM: {llm_loader.config.get('managerLLM')}")
        print(f"🔧 Available LLMs: {', '.join(llm_loader.list_available_llms())}")
        
        # Supported formats
        processor = DocumentProcessor()
        print(f"📄 Supported Formats: {', '.join(processor.get_supported_extensions())}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def main():
    """Main CLI loop"""
    print_banner()
    
    try:
        # Initialize components
        print("Initializing system...")
        llm_loader = get_llm_loader()
        vector_store = get_vector_store()
        chatbot = get_chatbot()
        print("✓ System initialized successfully\n")
        
    except Exception as e:
        print(f"❌ Initialization error: {str(e)}")
        print("Please check your configuration and try again.")
        return
    
    print_menu()
    
    # Command mapping
    commands = {
        '1': upload_documents,
        'upload': upload_documents,
        '2': chat_mode,
        'chat': chat_mode,
        '3': search_documents,
        'search': search_documents,
        '4': view_history,
        'history': view_history,
        '5': clear_memory,
        'clear': clear_memory,
        '6': show_status,
        'status': show_status,
        '7': print_menu,
        'help': print_menu,
    }
    
    while True:
        try:
            command = input("\nEnter command: ").strip().lower()
            
            if command in ['8', 'exit', 'quit']:
                print("\nGoodbye! 👋")
                break
            
            if command in commands:
                commands[command]()
            else:
                print("❌ Unknown command. Type 'help' to see available commands.")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
