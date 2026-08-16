# Multi-Modal RAG Pipeline

A document-based retrieval-augmented generation pipeline that processes PDFs, extracts text, tables, and images, enriches them with AI summaries, stores vector embeddings in ChromaDB, and answers questions using a multimodal LLM.

## Features

- PDF parsing with `unstructured`
- Title-based chunking for structured document sections
- Table and image extraction from PDF pages
- AI-enhanced chunk summaries using NVIDIA endpoints
- ChromaDB vector store for semantic retrieval
- Final answer generation using multimodal LLM context
- Ready to run from a clean Python environment

## Project structure

```text
MultiMRAG/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── multimodal_rag_pipeline.py
├── main.py
├── docs/
├── dbv1/
├── dbv2/
└── rag_results.json
```

## Prerequisites

You need:

- Python 3.10+
- Tesseract OCR installed and available on PATH
- Poppler utilities installed and available on PATH for PDF rendering if needed
- An NVIDIA API key with access to the NVIDIA inference and embedding models

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy the environment file:

```bash
copy .env.example .env
```

5. Fill in your API keys in `.env`:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

6. Ensure Tesseract and Poppler are installed and available in PATH.

## Run the pipeline

```bash
python main.py --pdf "docs/your_document.pdf" --query "What is this document about?"
```

Or run the module directly:

```bash
python multimodal_rag_pipeline.py --pdf "docs/your_document.pdf" --query "What is this document about?"
```

## Notes

This pipeline uses NVIDIA-hosted models for both embeddings and LLM generation. If you want to adapt it to another provider, update the model names in the code and the API configuration.

For Windows setup help, see the existing project guides for Tesseract and Poppler installation in this repository.

## Important

Do not commit your `.env` file or generated database directories to GitHub. They are ignored by default in `.gitignore`.
