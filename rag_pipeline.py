"""
Core RAG pipeline.
Handles: document loading → chunking → embedding → vector DB → retrieval → LLM answer
"""

from dotenv import load_dotenv
load_dotenv()

import os
import PyPDF2
from typing import List, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
import groq
from config import (
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
    VECTOR_DB_PATH, COLLECTION_NAME, LLM_MODEL, LLM_TEMPERATURE, SYSTEM_PROMPT
)


class DocumentProcessor:
    """Load and chunk documents."""

    @staticmethod
    def load_pdf(file_path: str) -> str:
        text = ""
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text

    @staticmethod
    def load_text(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading text file: {e}")
            return ""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 10:
                chunks.append(chunk)
        return chunks


class EmbeddingStore:
    """Manage embeddings and vector DB."""

    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✓ Embedding model loaded | Dimension: {self.embedding_model.get_sentence_embedding_dimension()}")

    def add_documents(self, chunks: List[str], source: str = "document") -> int:
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True).tolist()
        ids = [f"{source}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
        self.collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        print(f"✓ Added {len(chunks)} chunks to vector DB")
        return len(chunks)

    def retrieve(self, query: str, top_k: int = TOP_K, source_filter: str = None) -> List[Tuple[str, float]]:
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        where = {"source": source_filter} if source_filter else None
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)
        chunks = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        similarities = [1 - d for d in distances]
        return list(zip(chunks, similarities))

    def get_collection_stats(self) -> dict:
        count = self.collection.count() if self.collection else 0
        return {"total_chunks": count, "model": EMBEDDING_MODEL}

    def clear_collection(self):
        """Drop all documents from the collection."""
        try:
            self.client.delete_collection(name=COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print("✓ Collection cleared")
        except Exception as e:
            print(f"Error clearing collection: {e}")


class LLMEngine:
    """Call the LLM with context."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set.")
        self.client = groq.Groq(api_key=api_key)
        self.model = LLM_MODEL

    def generate_answer(self, context: str, query: str) -> str:
        messages = [{
            "role": "user",
            "content": f"""{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        }]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=500
        )
        return response.choices[0].message.content


class RAGPipeline:
    """Main RAG system."""

    def __init__(self):
        self.processor = DocumentProcessor()
        self.embedding_store = EmbeddingStore()
        self.llm_engine = LLMEngine()

    def index_document(self, file_path: str) -> dict:
        if file_path.endswith('.pdf'):
            text = self.processor.load_pdf(file_path)
        else:
            text = self.processor.load_text(file_path)
        if not text.strip():
            return {"status": "error", "message": "Could not extract text"}
        chunks = self.processor.chunk_text(text)
        source_name = os.path.basename(file_path)
        chunk_count = self.embedding_store.add_documents(chunks, source=source_name)
        return {"status": "success", "file": source_name, "chunks_created": len(chunks), "chunks_indexed": chunk_count}

    def query(self, question: str, source_document: str = None) -> dict:
        retrieved = self.embedding_store.retrieve(question, top_k=TOP_K, source_filter=source_document)
        if not retrieved:
            return {"answer": "No relevant information found.", "retrieved_chunks": [], "status": "no_context"}
        context = "\n".join([f"[Source {i}, similarity: {score:.2f}]\n{chunk}\n"
                             for i, (chunk, score) in enumerate(retrieved, 1)])
        answer = self.llm_engine.generate_answer(context, question)
        return {"answer": answer,
                "retrieved_chunks": [{"text": chunk[:200] + "...", "similarity": score} for chunk, score in retrieved],
                "status": "success"}

    def get_stats(self) -> dict:
        return self.embedding_store.get_collection_stats()

    def clear(self):
        self.embedding_store.clear_collection()
    def delete_document(self, source: str) -> dict:
     """
     Delete a specific document from the vector database.
    
     Args:
        source: Document name to delete (e.g., "test_paper.txt")
    
     Returns:
        Status of deletion
     """
     try:
        # Get all IDs with this source in metadata
        results = self.embedding_store.collection.get(
            where={"source": source}
        )
        
        if not results['ids']:
            return {
                "status": "not_found",
                "message": f"No documents found with source: {source}"
            }
        
        # Delete all chunks with this source
        self.embedding_store.collection.delete(
            ids=results['ids']
        )
        
        chunks_deleted = len(results['ids'])
        
        print(f"✓ Deleted {chunks_deleted} chunks from {source}")
        
        return {
            "status": "success",
            "message": f"Deleted {chunks_deleted} chunks from {source}",
            "chunks_deleted": chunks_deleted
        }
    
     except Exception as e:
        return {
            "status": "error",
            "message": f"Error deleting document: {str(e)}"
        }























































# """
# Core RAG pipeline.
# Handles: document loading → chunking → embedding → vector DB → retrieval → LLM answer
# """

# from dotenv import load_dotenv
# load_dotenv()


# import os
# import PyPDF2
# from typing import List, Tuple
# import chromadb
# from sentence_transformers import SentenceTransformer
# import groq
# from config import (
#     EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
#     VECTOR_DB_PATH, COLLECTION_NAME, LLM_MODEL, LLM_TEMPERATURE, SYSTEM_PROMPT
# )


# class DocumentProcessor:
#     """Load and chunk documents."""
    
#     @staticmethod
#     def load_pdf(file_path: str) -> str:
#         """Extract text from PDF."""
#         text = ""
#         try:
#             with open(file_path, 'rb') as f:
#                 reader = PyPDF2.PdfReader(f)
#                 for page_num in range(len(reader.pages)):
#                     page = reader.pages[page_num]
#                     text += page.extract_text() + "\n"
#         except Exception as e:
#             print(f"Error reading PDF: {e}")
#         return text
    
#     @staticmethod
#     def load_text(file_path: str) -> str:
#         """Load plain text file."""
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 return f.read()
#         except Exception as e:
#             print(f"Error reading text file: {e}")
#             return ""
    
#     @staticmethod
#     def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
#         """
#         Simple sliding window chunking.
#         Split by words (not tokens, simpler for students).
#         """
#         words = text.split()
#         chunks = []
        
#         for i in range(0, len(words), chunk_size - overlap):
#             chunk = " ".join(words[i:i + chunk_size])
#             if len(chunk.strip()) > 10:  # Skip tiny chunks
#                 chunks.append(chunk)
        
#         return chunks


# class EmbeddingStore:
#     """Manage embeddings and vector DB."""
    
#     def __init__(self):
#         """Initialize embedding model and vector DB."""
#         print(f"Loading embedding model: {EMBEDDING_MODEL}...")
#         self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
#         # Initialize ChromaDB
#         self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
#         self.collection = self.client.get_or_create_collection(
#             name=COLLECTION_NAME,
#             metadata={"hnsw:space": "cosine"}
#         )
        
#         print(f"✓ Embedding model loaded | Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
    
#     def add_documents(self, chunks: List[str], source: str = "document") -> int:
#         """Add chunks to vector DB with embeddings."""
#         print(f"Generating embeddings for {len(chunks)} chunks...")
        
#         # Generate embeddings
#         embeddings = self.embedding_model.encode(chunks, show_progress_bar=True).tolist()
        
#         # Add to DB
#         ids = [f"{source}_{i}" for i in range(len(chunks))]
#         metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
        
#         self.collection.add(
#             ids=ids,
#             documents=chunks,
#             embeddings=embeddings,
#             metadatas=metadatas
#         )
        
#         print(f"✓ Added {len(chunks)} chunks to vector DB")
#         return len(chunks)
    
#     def retrieve(self, query: str, top_k: int = TOP_K, source_filter: str = None) -> List[Tuple[str, float]]:
#         """
#         Retrieve top-k relevant chunks for a query.
#         Can filter by document source if provided.
    
#         Args:
#             query: User's question
#             top_k: Number of results to return
#             source_filter: Optional - only search chunks from this document
    
#         Returns: [(chunk_text, similarity_score), ...]
#         """
    
#         query_embedding = self.embedding_model.encode([query])[0].tolist()
    
#         # If source_filter provided, search only in that document
#         if source_filter:
#             # Use ChromaDB's where filter
#             results = self.collection.query(
#                 query_embeddings=[query_embedding],
#                 n_results=top_k,
#                 where={"source": source_filter}  # Filter by source!
#             )
#         else:
#             # Search all documents (default behavior)
#             results = self.collection.query(
#                 query_embeddings=[query_embedding],
#                 n_results=top_k
#             )
    
#         chunks = results['documents'][0] if results['documents'] else []
#         distances = results['distances'][0] if results['distances'] else []
    
#         similarities = [1 - d for d in distances]
    
#         return list(zip(chunks, similarities))
    
#     def get_collection_stats(self) -> dict:
#         """Get info about the vector DB."""
#         count = self.collection.count()
#         return {"total_chunks": count, "model": EMBEDDING_MODEL}


# class LLMEngine:
#     """Call the LLM with context."""
    
#     def __init__(self):
#         """Initialize Groq client."""
#         api_key = os.getenv("GROQ_API_KEY")
#         if not api_key:
#             raise ValueError("GROQ_API_KEY not set. Get it from https://console.groq.com/")
        
#         self.client = groq.Groq(api_key=api_key)
#         self.model = LLM_MODEL
    
#     def generate_answer(self, context: str, query: str) -> str:
#         """
#         Generate answer using retrieved context.
        
#         Args:
#             context: Retrieved chunks formatted as context
#             query: User's question
        
#         Returns:
#             LLM-generated answer
#         """
#         messages = [
#             {
#                 "role": "user",
#                 "content": f"""{SYSTEM_PROMPT}

# CONTEXT:
# {context}

# QUESTION: {query}

# ANSWER:"""
#             }
#         ]
        
#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=messages,
#             temperature=LLM_TEMPERATURE,
#             max_tokens=500
#         )
        
#         return response.choices[0].message.content


# class RAGPipeline:
#     """Main RAG system: tie everything together."""
    
#     def __init__(self):
#         """Initialize all components."""
#         self.processor = DocumentProcessor()
#         self.embedding_store = EmbeddingStore()
#         self.llm_engine = LLMEngine()
    
#     def index_document(self, file_path: str) -> dict:
#         """
#         Load a document and index it.
        
#         Returns:
#             Status dict with info
#         """
#         print(f"\n📄 Indexing: {file_path}")
        
#         # Load document
#         if file_path.endswith('.pdf'):
#             text = self.processor.load_pdf(file_path)
#         else:
#             text = self.processor.load_text(file_path)
        
#         if not text.strip():
#             return {"status": "error", "message": "Could not extract text"}
        
#         # Chunk
#         chunks = self.processor.chunk_text(text)
#         print(f"✓ Created {len(chunks)} chunks")
        
#         # Index
#         source_name = os.path.basename(file_path)
#         chunk_count = self.embedding_store.add_documents(chunks, source=source_name)
        
#         return {
#             "status": "success",
#             "file": source_name,
#             "chunks_created": len(chunks),
#             "chunks_indexed": chunk_count
#         }
    
#     def query(self, question: str, source_document: str = None) -> dict:
#         """
#         Full RAG pipeline with optional document filtering.
    
#         Args:
#             question: User's question
#             source_document: Optional - search only in this document
    
#         Returns:
#             Answer with retrieved chunks
#         """
#         print(f"❓ Query: {question}")
    
#         # Retrieve relevant chunks (optionally filtered by document)
#         retrieved = self.embedding_store.retrieve(
#             question,
#             top_k=TOP_K,
#             source_filter=source_document  # Pass the document filter!
#         )
    
#         if not retrieved:
#             return {
#                 "answer": "No relevant information found in the knowledge base.",
#                 "retrieved_chunks": [],
#                 "status": "no_context"
#             }
    
#         # Build context from retrieved chunks
#         context_parts = []
#         for i, (chunk, score) in enumerate(retrieved, 1):
#             context_parts.append(f"[Source {i}, similarity: {score:.2f}]\n{chunk}\n")
    
#         context = "\n".join(context_parts)
    
#         # Generate answer
#         print(f"✓ Retrieved {len(retrieved)} chunks, calling LLM...")
#         answer = self.llm_engine.generate_answer(context, question)
    
#         return {
#             "answer": answer,
#             "retrieved_chunks": [
#                 {"text": chunk[:200] + "...", "similarity": score}
#                 for chunk, score in retrieved
#             ],
#             "status": "success"
#         }
    
#     def get_stats(self) -> dict:
#         """Get system statistics."""
#         return self.embedding_store.get_collection_stats()

# # Quick test
# if __name__ == "__main__":
#     print("Testing RAG pipeline...\n")
    
#     rag = RAGPipeline()
#     print("✓ Pipeline initialized")
#     print(f"Stats: {rag.get_stats()}")
