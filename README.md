# Agentic RAG Assistant

An AI-powered document Q&A system that lets you upload PDF documents and ask questions about them. Built with LlamaIndex, Groq, and Gradio.

## Features

- **Multi-Document Support**: Upload up to 5 PDF documents
- **Agentic RAG**: Uses FunctionAgent with dynamic tool selection
- **Per-Document Tools**: Each document gets vector search + summary tools
- **Persistent Storage**: Indices are cached to disk for fast startup
- **Free & Fast**: Uses Groq (Llama 4 Scout 17B) + local HuggingFace embeddings

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   Tool Retriever    │  ← Selects top-k relevant tools
│   (ObjectIndex)     │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   FunctionAgent     │  ← Executes selected tools
│   (Groq LLM)        │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Document Tools     │
│  - vector_doc1      │  ← Semantic search
│  - summary_doc1     │  ← Summarization
│  - vector_doc2      │
│  - ...              │
└─────────────────────┘
    │
    ▼
   Response
```

## Tools

- **Vector Search Tool**: Retrieves specific information from documents using semantic similarity
- **Summary Tool**: Generates summaries of entire documents using tree summarization

## Models Used

- **LLM**: Groq Llama 4 Scout 17B (free, fast inference, higher rate limits)
- **Embeddings**: HuggingFace BGE-small-en-v1.5 (local, no API needed)
- **Framework**: LlamaIndex with FunctionAgent

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cloudchristina/Agentic-RAG.git
cd Agentic-RAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

4. Get a free Groq API key:
   - Visit https://console.groq.com/keys
   - Create an account and generate an API key

## Usage

1. Run the application:
```bash
python app.py
```

2. Open http://127.0.0.1:7860 in your browser

3. Upload PDF documents (max 5) and start asking questions!

### Example Queries

- "What is the main contribution of this paper?"
- "Summarize the methodology section"
- "Compare the approaches in document A and document B"

## Project Structure

```
agentic_rag/
├── app.py              # Gradio UI application
├── agent.py            # FunctionAgent setup
├── indexer.py          # Indexing and tool creation
├── helper.py           # Environment utilities
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── data/               # Uploaded PDF documents
└── storage/            # Persisted document indices
```

## How It Works

1. **Document Processing**: PDFs are loaded and split into chunks (1024 tokens)
2. **Index Creation**: Each document gets a VectorStoreIndex and SummaryIndex
3. **Tool Generation**: Two tools per document (vector search + summary)
4. **Tool Index**: ObjectIndex enables dynamic tool selection based on query
5. **Agent Execution**: FunctionAgent uses relevant tools to answer queries

## Test

Download sample PDFs to test the application:

```bash
mkdir -p test/
wget -O test/metagpt.pdf "https://openreview.net/pdf?id=VtmBAGCN7o"
wget -O test/longlora.pdf "https://openreview.net/pdf?id=6PmJoRfdaK"
wget -O test/loftq.pdf "https://openreview.net/pdf?id=LzPWWPAdY4"
wget -O test/swebench.pdf "https://openreview.net/pdf?id=VTF8yNQM66"
wget -O test/selfrag.pdf "https://openreview.net/pdf?id=hSyW5go0v8"
# wget -O test/zipformer.pdf "https://openreview.net/pdf?id=9WD9KwssyT"
# wget -O test/values.pdf "https://openreview.net/pdf?id=yV6fD7LYkF"
# wget -O test/finetune_fair_diffusion.pdf "https://openreview.net/pdf?id=hnrB5YHoYu"
# wget -O test/knowledge_card.pdf "https://openreview.net/pdf?id=WbWtOYIzIK"
# wget -O test/metra.pdf "https://openreview.net/pdf?id=c5pwL0Soay"
# wget -O test/vr_mcl.pdf "https://openreview.net/pdf?id=TpD2aG1h0D"
```

Then run `python app.py` and try queries like:
- "What is MetaGPT?"
- "Summarize zipformer"
- "Tell me about the evaluation dataset used in MetaGPT and compare it against longlora"

## Requirements

- Python 3.10+
- Groq API key (free tier available)
- ~500MB disk space for embeddings model
