# 🚀 Quick Start Guide

Get the RAG system running in 5 minutes.

## Step 1: Get API Key 

1. Go to https://console.groq.com/ (for GROQ API key you can use Gemini API Key as well)
2. Sign up (free)
3. Copy your API key

## Step 2: Setup Project

```bash
# Clone or download project
cd Document_Intelligence

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export GROQ_API_KEY="your-api-key-from-step-1"
# Or on Windows:
set GROQ_API_KEY=your-api-key-from-step-1
```

## Step 3: Run App

```bash
streamlit run app.py
```

Then:
- Browser opens to http://localhost:8501
- Upload a PDF or text file
- Click "Index Document"
- Ask a question!


### "Might be Slow on first run"
- Normal! The embedding model (130MB) downloads on first use
- Takes 30-60 seconds, then it's cached

### Want to test without Streamlit?
```bash
python test_rag.py
```

## Next Steps

1. ✅ Run the app
2. ✅ Upload a document (try a PDF or text file)
3. ✅ Ask it questions
4. 📚 Read README.md to understand what's happening
5. 💻 You can modify config.py to experiment with different settings

---
