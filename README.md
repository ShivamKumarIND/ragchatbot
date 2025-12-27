# RAG-Based AI Chatbot

A powerful document-based Question & Answer system using Retrieval Augmented Generation (RAG), LangChain, and Groq LLM.

## Features

- 📄 **Multiple Document Formats**: Supports PDF, DOCX, HTML, CSV, XLSX, and TXT files
- 🧠 **Conversation Memory**: Remembers context from previous messages using LangChain memory
- 🔍 **Vector Search**: Fast semantic search using ChromaDB and HuggingFace embeddings
- ⚙️ **Configurable LLMs**: Easy LLM switching via JSON configuration
- 🚀 **REST API**: FastAPI-based REST API for integration
- 💻 **CLI Interface**: Simple command-line interface for local usage
- 📊 **Source Attribution**: Shows which documents were used to answer questions

## Project Structure

```
ragchatbot/
├── app.py                    # FastAPI REST API server
├── cli.py                    # Command-line interface
├── llm_loader.py            # LLM configuration loader
├── document_processor.py     # Document loading and processing
├── vector_store.py          # Vector store management
├── rag_chain.py             # RAG chain with memory
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
├── .gitignore              # Git ignore rules
├── config/
│   └── llm.json            # LLM configuration
├── uploads/                # Uploaded documents (auto-created)
└── chroma_db/              # Vector database (auto-created)
```

## Installation

### 1. Clone or Setup

```bash
cd c:\ragchatbot
```

### 2. Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Environment

Edit `.env` file and add your Groq API key:

```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

Get your API key from: https://console.groq.com/

## Configuration

### LLM Configuration (`config/llm.json`)

The system uses `llm.json` for LLM configuration. You can add multiple LLMs:

```json
{
    "llms": {
        "Groq-Llama": {
            "display_name": "Groq Llama 3.1 70B",
            "import_module": "langchain_groq",
            "import_class": "ChatGroq",
            "load_on_init": "True",
            "max_input_chars": 120000,
            "config": {
                "model_name": "llama-3.1-70b-versatile",
                "groq_api_key": "ENV:GROQ_API_KEY",
                "temperature": 0.0,
                "max_tokens": 2048,
                "streaming": true
            }
        }
    },
    "managerLLM": "Groq-Llama"
}
```

- `ENV:VARIABLE_NAME` references environment variables from `.env`
- `managerLLM` specifies which LLM to use
- `load_on_init` controls automatic loading on startup

## Usage

### Option 1: Command Line Interface (CLI)

```powershell
python cli.py
```

Available commands:
- `upload` - Upload and process documents
- `chat` - Start chatting with the assistant
- `search` - Search for relevant documents
- `history` - View chat history
- `clear` - Clear conversation memory
- `status` - View system status
- `exit` - Exit the application

### Option 2: REST API Server

Start the server:

```powershell
python app.py
```

Server will run at: `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

#### API Endpoints

**1. Upload Documents**
```bash
POST /upload
Content-Type: multipart/form-data

files: [file1.pdf, file2.docx, ...]
```

**2. Chat**
```bash
POST /chat
Content-Type: application/json

{
    "question": "What is this document about?",
    "session_id": "optional-session-id"
}
```

**3. Get Status**
```bash
GET /status
```

**4. Search Documents**
```bash
GET /search?query=your+search+query&k=4
```

**5. Get Chat History**
```bash
GET /history
```

**6. Clear Memory**
```bash
POST /clear
```

**7. Delete All Documents**
```bash
DELETE /documents
```

## Example Usage

### Python Script Example

```python
from document_processor import DocumentProcessor
from vector_store import get_vector_store
from rag_chain import get_chatbot

# Process documents
processor = DocumentProcessor()
docs = processor.process_document("path/to/document.pdf")

# Add to vector store
vector_store = get_vector_store()
vector_store.add_documents(docs)

# Chat with the bot
chatbot = get_chatbot()
chatbot.reinitialize_chain()

response = chatbot.chat("What is the main topic of this document?")
print(response["answer"])
```

### cURL Example

```bash
# Upload a document
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document.pdf"

# Ask a question
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

## Supported Document Formats

- **PDF** (.pdf)
- **Word** (.docx, .doc)
- **HTML** (.html, .htm)
- **CSV** (.csv)
- **Excel** (.xlsx, .xls)
- **Text** (.txt)

## Architecture

### Components

1. **LLM Loader** (`llm_loader.py`)
   - Loads LLM configuration from `llm.json`
   - Manages multiple LLM instances
   - Resolves environment variables

2. **Document Processor** (`document_processor.py`)
   - Loads various document formats
   - Splits documents into chunks
   - Preserves metadata

3. **Vector Store** (`vector_store.py`)
   - Uses ChromaDB for storage
   - HuggingFace embeddings (all-MiniLM-L6-v2)
   - Similarity search

4. **RAG Chain** (`rag_chain.py`)
   - ConversationalRetrievalChain
   - Conversation memory buffer
   - Custom prompts for better answers

5. **API Server** (`app.py`)
   - FastAPI REST API
   - Document upload
   - Chat endpoints

## Memory Management

The system uses LangChain's `ConversationBufferMemory` to:
- Store chat history
- Maintain context across conversations
- Provide coherent multi-turn conversations

Memory can be cleared using:
- CLI: `clear` command
- API: `POST /clear`
- Code: `chatbot.clear_memory()`

## Advanced Configuration

### Changing Chunk Size

Edit `.env`:
```env
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
```

### Using Different Embeddings

Edit `vector_store.py` to change the embedding model:
```python
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### Adding New LLMs

Add to `config/llm.json`:
```json
{
    "llms": {
        "OpenAI-GPT4": {
            "display_name": "OpenAI GPT-4",
            "import_module": "langchain_openai",
            "import_class": "ChatOpenAI",
            "load_on_init": "False",
            "config": {
                "model_name": "gpt-4",
                "openai_api_key": "ENV:OPENAI_API_KEY",
                "temperature": 0.0
            }
        }
    }
}
```

## Troubleshooting

### Issue: "Vector store is empty"
**Solution**: Upload documents first using the upload command or API endpoint.

### Issue: "Environment variable not found"
**Solution**: Check that your `.env` file contains the required API keys.

### Issue: "Module not found"
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

### Issue: ChromaDB errors
**Solution**: Delete the `chroma_db` folder and re-upload documents.

## Performance Tips

1. **Batch Upload**: Upload multiple documents at once for efficiency
2. **Chunk Size**: Adjust based on your document type (larger for technical docs)
3. **Number of Results (k)**: Increase for more context, decrease for faster responses
4. **Clear Memory**: Clear memory periodically for long conversations

## Security Notes

- Store API keys in `.env` file (never commit to version control)
- `.gitignore` is configured to exclude sensitive files
- For production, configure CORS properly in `app.py`
- Use environment-specific configurations

## Dependencies

Key packages:
- `langchain` & `langchain-community`: RAG framework
- `langchain-groq`: Groq LLM integration
- `chromadb`: Vector database
- `sentence-transformers`: Embeddings
- `fastapi`: REST API framework
- `pypdf`, `python-docx`, `openpyxl`: Document loaders

See `requirements.txt` for complete list with versions.

## Contributing

Feel free to extend this project by:
- Adding new document loaders
- Implementing streaming responses
- Adding authentication
- Creating a web UI
- Supporting more LLM providers

## License

This project is provided as-is for educational and commercial use.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the API documentation at `/docs`
3. Check LangChain documentation: https://python.langchain.com/

---

**Built with ❤️ using LangChain, Groq, and FastAPI**
