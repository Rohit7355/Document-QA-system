"""
Quick test of RAG pipeline without Streamlit.
Use this to test the core functionality.

Usage:
    python test_rag.py
"""

import os
from pathlib import Path
from rag_pipeline import RAGPipeline

def create_sample_document():
    """Create a sample document for testing."""
    sample_text = """
    Deep Learning is a subset of Machine Learning that uses neural networks with multiple layers.
    
    Deep Learning has revolutionized artificial intelligence in recent years. It powers applications
    like image recognition, natural language processing, and game-playing AI.
    
    A neural network is inspired by biological neurons in the brain. It consists of layers of
    interconnected nodes that process information. The "deep" in deep learning refers to the
    multiple layers in these networks.
    
    Common architectures include:
    - Convolutional Neural Networks (CNNs): Good for image processing
    - Recurrent Neural Networks (RNNs): Good for sequential data like text
    - Transformers: State-of-the-art for language tasks
    
    Training a deep learning model requires:
    1. A large dataset
    2. Computational resources (GPUs)
    3. Careful hyperparameter tuning
    4. Proper regularization to prevent overfitting
    
    The field of deep learning has grown exponentially with the availability of large datasets
    (ImageNet, Wikipedia) and powerful GPUs (NVIDIA GPUs).
    """
    
    with open("sample.txt", "w") as f:
        f.write(sample_text)
    
    return "sample.txt"

def test_pipeline():
    """Test the full RAG pipeline."""
    print("=" * 60)
    print("🤖 RAG Pipeline Test")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        print("\n❌ ERROR: GROQ_API_KEY not set!")
        print("Please set your API key:")
        print("  export GROQ_API_KEY='your-key-here'")
        print("\nGet a free key at: https://console.groq.com/")
        return
    
    print("\n✓ API key configured\n")
    
    # Initialize
    print("1️⃣  Initializing RAG Pipeline...")
    rag = RAGPipeline()
    print("   ✓ Pipeline ready\n")
    
    # Check if we have documents already
    stats = rag.get_stats()
    print(f"Current state: {stats['total_chunks']} chunks indexed\n")
    
    if stats["total_chunks"] == 0:
        # Create and index sample document
        print("2️⃣  Creating sample document...")
        doc_path = create_sample_document()
        print(f"   ✓ Created {doc_path}\n")
        
        print("3️⃣  Indexing document...")
        result = rag.index_document(doc_path)
        print(f"   ✓ {result['status'].upper()}: {result['chunks_created']} chunks created\n")
    
    # Test queries
    test_questions = [
        "What is deep learning?",
        "What are the main neural network architectures?",
        "How do you train a deep learning model?",
    ]
    
    print("4️⃣  Running test queries...")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📌 Query {i}: {question}")
        print("-" * 60)
        
        result = rag.query(question)
        
        # Show answer
        print(f"\n💡 Answer:\n{result['answer']}\n")
        
        # Show retrieved chunks
        print(f"📍 Retrieved {len(result['retrieved_chunks'])} chunks:")
        for j, chunk in enumerate(result['retrieved_chunks'], 1):
            print(f"   [{j}] Similarity: {chunk['similarity']:.1%}")
            print(f"       {chunk['text'][:100]}...")
        
        print("-" * 60)
    
    # Cleanup
    if Path("sample.txt").exists():
        Path("sample.txt").unlink()
    
    print("\n✅ Test complete!")
    print("\nNext steps:")
    print("  1. Run Streamlit app: streamlit run app.py")
    print("  2. Upload your own documents")
    print("  3. Try different questions")

if __name__ == "__main__":
    test_pipeline()
