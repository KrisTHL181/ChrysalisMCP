import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger

# --- Configuration ---
RESOURCES_DIR = "mcp_resources"
INDEX_PATH = "faiss_index"
CHUNKS_PATH = "chunks.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@logger.catch
def build_index():
    """Builds a FAISS index from documents in the resources directory."""
    logger.info(f"Starting to build index from documents in: {RESOURCES_DIR}")

    # Load documents
    loader = DirectoryLoader(
        RESOURCES_DIR,
        loader_cls=TextLoader, # Using TextLoader for .txt files
        glob="**/*.txt",
        show_progress=True,
        use_multithreading=True,
    )
    documents = loader.load()
    if not documents:
        logger.warning("No documents found to index. Exiting.")
        return

    logger.info(f"Loaded {len(documents)} documents.")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks.")

    # Create embeddings
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Build FAISS index
    logger.info("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    logger.info(f"FAISS index saved to: {INDEX_PATH}")

    # Save chunk metadata
    chunk_data = [
        {"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks
    ]
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=4)
    logger.info(f"Chunk metadata saved to: {CHUNKS_PATH}")

    logger.info("Index building complete.")

if __name__ == "__main__":
    build_index()
