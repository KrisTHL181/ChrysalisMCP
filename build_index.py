import glob
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_core.documents import Document
import pytesseract
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from loguru import logger
import default_api as default_api_module

# --- Configuration ---
RESOURCES_DIR = "mcp_resources"
INDEX_PATH = "faiss_index"
CHUNKS_PATH = "chunks.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class ImageLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        try:
            text = pytesseract.image_to_string(Image.open(self.file_path))
            return [Document(page_content=text, metadata={"source": self.file_path, "type": "image"})]
        except Exception as e:
            logger.error(f"Error processing image {self.file_path}: {e}")
            return []

@logger.catch
def build_index():
    """Builds a FAISS index from documents in the resources directory."""
    logger.info(f"Starting to build index from documents in: {RESOURCES_DIR}")

    # Load documents
    documents = []
    
    # Load .txt files
    txt_loader = DirectoryLoader(
        RESOURCES_DIR,
        loader_cls=TextLoader,
        glob="**/*.txt",
        show_progress=True,
        use_multithreading=True,
    )
    documents.extend(txt_loader.load())

    # Load .pdf files
    pdf_loader = DirectoryLoader(
        RESOURCES_DIR,
        loader_cls=PyPDFLoader,
        glob="**/*.pdf",
        show_progress=True,
        use_multithreading=True,
    )
    documents.extend(pdf_loader.load())

    # Load .docx files
    docx_loader = DirectoryLoader(
        RESOURCES_DIR,
        loader_cls=Docx2txtLoader,
        glob="**/*.docx",
        show_progress=True,
        use_multithreading=True,
    )
    documents.extend(docx_loader.load())

    # Load .pptx files (requires unstructured and its dependencies)
    # Note: UnstructuredPowerPointLoader might require additional system dependencies
    # like libmagic and poppler-utils. For simplicity, we'll use it directly here.
    # If you encounter errors, you might need to install these system-wide.
    pptx_loader = DirectoryLoader(
        RESOURCES_DIR,
        loader_cls=UnstructuredPowerPointLoader,
        glob="**/*.pptx",
        show_progress=True,
        use_multithreading=True,
    )
    documents.extend(pptx_loader.load())

    # Load image files (PNG, JPG, JPEG) using OCR
    image_files = []
    for ext in ["png", "jpg", "jpeg"]:
        image_files.extend(glob.glob(os.path.join(RESOURCES_DIR, f"**/*.{ext}"), recursive=True))

    for img_file in image_files:
        loader = ImageLoader(img_file)
        documents.extend(loader.load())

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

    # Update global memory after index is built
    logger.info("Updating global memory...")
    # User can customize the summary model by passing it as an argument to build_index.py
    # For example: python build_index.py --summary-model "facebook/bart-large-cnn"
    # For now, we use a default model.
    default_api_module.update_global_memory(summary_model_name="sshleifer/distilbart-cnn-12-6")
    logger.info("Global memory update complete.")

if __name__ == "__main__":
    build_index()
