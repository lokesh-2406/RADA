# RADA - RAG-Assisted Document Analysis

RADA is a powerful RAG (Retrieval-Augmented Generation) application for analyzing PDF documents. Upload your documents and ask questions to get insights from their content.

## Features

- **PDF Processing**: Upload and process multiple PDF documents
- **Intelligent Chunking**: Three high-quality document chunking strategies
- **Vector Storage**: Efficiently store and search document chunks
- **Conversational Interface**: Ask questions about your documents in a chat interface

## Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API Key (for embeddings and LLM)

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```bash
   python app.py
   ```

## Advanced Chunking Methods

RADA implements three industry-leading chunking strategies that can be selected based on your document type:

### 1. Recursive Chunking

**Best for**: General-purpose use, documents with varied content

This method splits documents hierarchically by paragraphs, then sentences, then characters. It's more context-aware than simple chunking and preserves the natural flow of documents better. The recursive approach ensures that related content stays together while still creating chunks of manageable size.

### 2. Semantic Chunking

**Best for**: Preserving semantic meaning, complex documents where context is crucial

This state-of-the-art method uses OpenAI embeddings to identify natural semantic breaks in the text. Instead of splitting arbitrarily by character count, it analyzes the content to find where topic changes occur. This approach significantly improves retrieval relevance by keeping semantically related content together.

### 3. Markdown Header Chunking

**Best for**: Technical documentation, research papers, organized content with clear sections

This specialized technique splits documents based on Markdown headers (like #, ##, ###), preserving the document's hierarchical structure. It's particularly effective for well-structured documents where sections naturally align with content topics. This method also preserves header information in metadata for improved context in responses.

## Chunking Parameters

- **Chunk Size**: Controls the target size of text chunks
  - Larger values (1000-2000): More context, but potentially less precise retrieval
  - Smaller values (250-500): More granular retrieval, but may lose context

- **Chunk Overlap**: Controls how much text overlaps between consecutive chunks
  - Higher values (200-500): Better preserves context across chunks
  - Lower values (50-100): Less redundancy, more efficient storage

## Usage Tips

- **Technical Documentation**: Choose Markdown Header chunking
- **Research Papers**: Try Semantic chunking with larger chunks
- **Books or Articles**: Recursive chunking works well with medium-sized chunks
- **Mixed Content**: Start with Recursive and experiment with others as needed

## Project Structure

```
rada/
├── app.py                   # Main application entry point
├── src/
│   ├── extract_from_pdf.py  # PDF loading and processing
│   ├── predict.py           # Question answering logic
│   ├── vectorstore.py       # Vector database management
│   └── chunking/            # Document chunking methods
│       ├── advanced_chunking.py # Implementation of chunking algorithms
│       └── chunking_factory.py # Factory for selecting chunking methods
├── chroma_db/               # Vector database storage
└── requirements.txt         # Project dependencies
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.