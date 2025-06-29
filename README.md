# ChrysalisMCP: A Model Community Platform (MCP) Project

ChrysalisMCP (Model Community Platform) is a powerful command-line interface (CLI) based application designed to serve as a foundational platform for integrating and enhancing Large Language Models (LLMs) with advanced capabilities. It focuses on providing a robust framework for model interaction, including Retrieval-Augmented Generation (RAG) and a persistent memory system, allowing LLMs to leverage local knowledge bases for more accurate and context-aware responses, simulating a "global memory" for long-context processing.

## Features

- **Hybrid Retrieval**: Combines semantic search (FAISS) and keyword search (BM25) for comprehensive document retrieval.
- **Multi-modal RAG**: Extracts text from various document types including:
  - Plain Text (`.txt`)
  - PDF (`.pdf`)
  - Microsoft Word (`.docx`)
  - Microsoft PowerPoint (`.pptx`)
  - Images (`.png`, `.jpg`, `.jpeg`) via OCR (requires Tesseract OCR engine).
- **LLM-Driven Clue Generation**: Enhances retrieval queries by generating more detailed "clues" using an LLM, based on the original query and global memory.
- **Global Memory**: Maintains a persistent, summarized overview of the entire knowledge base, automatically updated during indexing.
- **Persistent User Memory**: Saves specific user-related facts to long-term memory.
- **Extensible Prompt System**: Easily define and extend various LLM prompts (e.g., code generation, summarization, explanation).
- **Tool Integration**: Seamlessly integrates with external tools like Google Search, web fetching, and shell commands.
- **Modular Design**: Built with a clear separation of concerns for easy maintenance and extension.

## Setup

Follow these steps to set up and run ChrysalisMCP locally.

### 1. Clone the Repository

```bash
git clone https://github.com/KrisTHL181/ChrysalisMCP.git
cd ChrysalisMCP
```

### 2. Install Dependencies

Ensure you have Python (3.9+) and pip installed. Then, install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR (for Image Support)

If you plan to use image files (`.png`, `.jpg`, `.jpeg`) in your knowledge base, you **must** install the Tesseract OCR engine on your system. Follow the installation instructions for your operating system:

- **Windows**: [Tesseract-OCR for Windows](https://tesseract-ocr.github.io/tessdoc/Installation.html#windows)
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

### 4. Configure API Keys

ChrysalisMCP uses local summarization models for global memory summarization. By default, it uses `sshleifer/distilbart-cnn-12-6`. You can customize the model by passing `summary_model_name` argument to `build_index.py` (e.g., `python build_index.py --summary-model "facebook/bart-large-cnn"`).

If you plan to use Google Search tool, you need to configure `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_CX` in `default_api.py`.

### 5. Prepare Knowledge Base

Place your `.txt`, `.pdf`, `.docx`, `.pptx`, `.png`, `.jpg`, and `.jpeg` files into the `mcp_resources/` directory. These documents will be indexed and used by the RAG system.

### 6. Build the FAISS Index and Global Memory

Run the `build_index.py` script to process your documents, create the FAISS vector index, and generate the global memory summary. This step will take some time, especially for large knowledge bases or if downloading the embedding model for the first time.

```bash
python build_index.py
```

## Usage

### 1. Start the MCP Server

```bash
python mcp_server.py
```

The server will start on `http://0.0.0.0:8000` by default.

### 2. Interact with the CLI (Example: Using `ask-rag`)

Once the server is running, you can interact with it via the CLI. For example, to use the `ask-rag` prompt:

```bash
# Initial call to ask-rag (will prompt for clue generation)
# The CLI client will handle the interaction to get the clue from the LLM
# and then re-call ask-rag with the generated clue.
# Example (conceptual, actual CLI interaction may vary):
# user: ask-rag --query "What is the main topic of the document about RAG?"
# CLI: (internally calls LLM to generate clue)
# CLI: (re-calls ask-rag with generated clue)
# CLI: (displays answer)
```

**Note**: The current `ask-rag` implementation in `mcp_server.py` is designed for a client that can handle multi-step interactions (i.e., receiving a prompt for a clue, sending it to an LLM, and then re-calling `ask-rag` with the generated clue). A simple direct call from a basic CLI might not fully demonstrate this flow without a more sophisticated client.

## Project Structure

```
MCP/
├── default_api.py          # Core tool implementations (file system, web, memory, etc.)
├── mcp_calculator.py       # Calculator tool implementation
├── mcp_server.py           # Main server application, prompt definitions, and tool wrappers
├── build_index.py          # Script to build FAISS index and global memory
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore file
├── mcp_resources/          # Directory for your knowledge base documents
│   ├── your_document.txt
│   ├── another_doc.pdf
│   └── image_with_text.png
├── faiss_index/            # Generated FAISS index files (ignored by Git)
└── chunks.json             # Metadata for document chunks (ignored by Git)
```

## Contributing

Contributions are welcome! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the [MIT License](LICENSE).
