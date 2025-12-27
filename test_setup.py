"""
Quick test script to verify the RAG system setup
"""
import os
from pathlib import Path


def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from llm_loader import get_llm_loader
        from document_processor import DocumentProcessor
        from vector_store import get_vector_store
        from rag_chain import get_chatbot
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import error: {str(e)}")
        return False


def test_config_files():
    """Test if configuration files exist"""
    print("\nTesting configuration files...")
    
    files_to_check = [
        ".env",
        "config/llm.json",
        "requirements.txt"
    ]
    
    all_exist = True
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} not found")
            all_exist = False
    
    return all_exist


def test_env_variables():
    """Test if required environment variables are set"""
    print("\nTesting environment variables...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    groq_key = os.getenv("GROQ_API_KEY")
    
    if groq_key and groq_key != "your_groq_api_key_here":
        print("✓ GROQ_API_KEY is set")
        return True
    else:
        print("✗ GROQ_API_KEY not set or using placeholder")
        print("  Please update your .env file with a real API key")
        return False


def test_llm_loader():
    """Test LLM loader"""
    print("\nTesting LLM loader...")
    
    try:
        from llm_loader import get_llm_loader
        
        loader = get_llm_loader()
        available_llms = loader.list_available_llms()
        
        print(f"✓ LLM loader initialized")
        print(f"  Available LLMs: {', '.join(available_llms)}")
        print(f"  Manager LLM: {loader.config.get('managerLLM')}")
        
        return True
    except Exception as e:
        print(f"✗ LLM loader error: {str(e)}")
        return False


def test_document_processor():
    """Test document processor"""
    print("\nTesting document processor...")
    
    try:
        from document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        extensions = processor.get_supported_extensions()
        
        print(f"✓ Document processor initialized")
        print(f"  Supported formats: {', '.join(extensions)}")
        
        return True
    except Exception as e:
        print(f"✗ Document processor error: {str(e)}")
        return False


def test_vector_store():
    """Test vector store"""
    print("\nTesting vector store...")
    
    try:
        from vector_store import get_vector_store
        
        vector_store = get_vector_store()
        doc_count = vector_store.get_collection_count()
        
        print(f"✓ Vector store initialized")
        print(f"  Documents indexed: {doc_count}")
        
        return True
    except Exception as e:
        print(f"✗ Vector store error: {str(e)}")
        return False


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    
    print("\n1. Install dependencies (if not done):")
    print("   pip install -r requirements.txt")
    
    print("\n2. Set your Groq API key in .env file:")
    print("   GROQ_API_KEY=your_actual_api_key_here")
    print("   Get it from: https://console.groq.com/")
    
    print("\n3. Run the application:")
    print("   Option A - CLI:  python cli.py")
    print("   Option B - API:  python app.py")
    
    print("\n4. Upload documents and start chatting!")
    print()


def main():
    """Run all tests"""
    print("=" * 60)
    print("RAG CHATBOT - SYSTEM TEST")
    print("=" * 60)
    
    results = []
    
    results.append(("Configuration Files", test_config_files()))
    results.append(("Imports", test_imports()))
    results.append(("Environment Variables", test_env_variables()))
    results.append(("LLM Loader", test_llm_loader()))
    results.append(("Document Processor", test_document_processor()))
    results.append(("Vector Store", test_vector_store()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Your RAG system is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
    
    print_next_steps()


if __name__ == "__main__":
    main()
