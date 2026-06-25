# Document Intelligence: A RAG Q&A System 

A "Multi-Format Research Paper & Chart Analyzer" that parses complex, multi-modal documents like college textbooks or academic PDFs.
A clean, working Retrieval-Augmented Generation system. This project demonstrates understanding of modern AI concepts and practical system building skills. You can upload your documents and ask questions based on it. It will extract chunks from the documents and will answer the question accordingly.

## 🚀 Setup & Usage

### 1. Prerequisites
- Python 3.8+
- Free Groq API key (https://console.groq.com/) or you can use Gemini API Key as well.

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Set API Key
#  Create .env file or export environment variable
export GROQ_API_KEY="your-api-key-here"

### 4. Run the App
streamlit run app.py

Then:
1. Open http://localhost:8501 in browser
2. Upload a PDF or text file
3. Click "Index Document"
4. Ask questions!
5. You can clear document after asking questions by clicking "Delete" for specific deletion or "Clear ALL Documents" for all documents deletion.


### 5. Testing Locally (without Streamlit)
# I had created a demo file for testing "test_rag.py" for testing the model.
```python
from rag_pipeline import RAGPipeline

# Initialize
rag = RAGPipeline()

# Index a document
rag.index_document("example.pdf")

# Ask a question
result = rag.query("What is the main topic?")
print(result["answer"])
```

---

## 📁 Project Structure

```
rag_project/
├── config.py              # Configuration (models, chunk sizes, etc.)
├── rag_pipeline.py        # Core RAG logic (4 classes)
├── app.py                 # Streamlit UI
├── requirements.txt       # Dependencies
├── chroma_db/             # Vector DB storage (created automatically)
├── temp_*.pdf/txt         # Temporary uploaded files
└── README.md              # Getting an Overview of project
├── env.example            # For API Key generation
├── SETUP.md               # Details about how to run the project
├── test_rag.py            # A demo file created just for testing the system without streamlit
```