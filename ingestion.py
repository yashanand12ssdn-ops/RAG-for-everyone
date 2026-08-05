import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma


load_dotenv()


def get_embedding_model():
    """
    Create NVIDIA embedding model.

    Make sure your .env file contains:
    NVIDIA_API_KEY=your_nvidia_api_key_here
    """

    embedding_model = NVIDIAEmbeddings(
        model="nvidia/nemotron-3-embed-1b"
    )

    return embedding_model


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory."""

    print(f"Loading documents from {docs_path}...")

    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory {docs_path} does not exist. "
            "Please create it and add your company files."
        )

    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No .txt files found in {docs_path}. "
            "Please add your company documents."
        )

    # Show first 2 documents
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks."""

    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i + 1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print("Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store using NVIDIA embeddings."""

    print("Creating NVIDIA embeddings and storing in ChromaDB...")

    embedding_model = get_embedding_model()

    print("--- Creating vector store ---")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("--- Finished creating vector store ---")
    print(f"Vector store created and saved to {persist_directory}")

    return vectorstore


def load_existing_vector_store(persist_directory="db/chroma_db"):
    """Load an existing ChromaDB vector store using NVIDIA embeddings."""

    print("Loading existing vector store...")

    embedding_model = get_embedding_model()

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(
        f"Loaded existing vector store with "
        f"{vectorstore._collection.count()} documents"
    )

    return vectorstore


def main():
    """Main ingestion pipeline."""

    print("=== RAG Document Ingestion Pipeline ===\n")

    # Define paths
    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    # Check if vector store already exists
    if os.path.exists(persistent_directory):
        print("✅ Vector store already exists. No need to re-process documents.")

        vectorstore = load_existing_vector_store(
            persist_directory=persistent_directory
        )

        return vectorstore

    print("Persistent directory does not exist. Initializing vector store...\n")

    # Step 1: Load documents
    documents = load_documents(docs_path)

    # Step 2: Split documents into chunks
    chunks = split_documents(
        documents=documents,
        chunk_size=1000,
        chunk_overlap=0
    )

    # Step 3: Create vector store
    vectorstore = create_vector_store(
        chunks=chunks,
        persist_directory=persistent_directory
    )

    print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.")

    return vectorstore


if __name__ == "__main__":
    main()
