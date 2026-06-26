"""
Configuration file for RAG system.
Centralized settings for easy tweaking.
"""

# Model and API Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, lightweight (384 dims)
CHUNK_SIZE = 512  # tokens per chunk
CHUNK_OVERLAP = 50  # overlap between chunks
TOP_K = 5  # top-k documents to retrieve

# LLM Settings (using Groq )
LLM_API_KEY_ENV = "GROQ_API_KEY"  # or use "GEMINI_API_KEY"
LLM_MODEL = "llama-3.1-8b-instant"  # Fast open-source model
LLM_TEMPERATURE = 0.3  # Lower = more factual

# Vector DB Settings
VECTOR_DB_PATH = "./chroma_db"  # Local storage
COLLECTION_NAME = "documents"

# App Settings
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

# System Prompt (keeps LLM focused)
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided context.
Always cite which part of the context you're using.
If the context doesn't contain information to answer the question, say so clearly.
Keep answers concise but complete."""
