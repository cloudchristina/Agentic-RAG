import gradio as gr
import asyncio
from pathlib import Path
from typing import List, Tuple
import shutil

from indexer import (
    build_all_doc_tools,
    create_tool_index,
    get_tool_retriever,
    get_indexed_files,
    DATA_DIR,
    init_settings,
)
from agent import create_agent, chat

# Initialize LlamaIndex settings (embedding model, LLM)
init_settings()

# Global state
agent = None
tool_index = None

DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_pdf_files() -> List[Path]:
    """
    Get list of PDF files in data directory.

    Returns:
        List[Path]: List of PDF file paths.
    """
    return list(DATA_DIR.glob("*.pdf"))

def initialize_agent():
    """
    Initialize or reinitialize the agent with current documents.

    Returns:
        str: Status message.
    """
    global agent, tool_index

    pdf_files = get_pdf_files()
    if not pdf_files:
        agent = None
        tool_index = None
        return "No documents indexed. Please upload PDFs."

    pdf_paths = [str(p) for p in pdf_files]
    all_tools = build_all_doc_tools(pdf_paths)
    tool_index = create_tool_index(all_tools)
    retriever = get_tool_retriever(tool_index, top_k=2)
    agent = create_agent(retriever)

    indexed = get_indexed_files()
    return f"Indexed {len(indexed)} documents: {', '.join(indexed)}"

def upload_files(files) -> str:
    """
    Handle file uploads.

    Args:
        files: List of uploaded files.
    Returns:
        str: Status message.
    """
    if not files:
        return get_status()

    # Check limit
    existing = list(DATA_DIR.glob("*.pdf"))
    if len(existing) + len(files) > 5:
        return f"Error: Maximum 5 documents allowed. Currently have {len(existing)}."

    # Save uploaded files
    for file in files:
        dest = DATA_DIR / Path(file.name).name
        shutil.copy(file.name, dest)

    # Reinitialize agent
    status = initialize_agent()
    return status

def clear_documents() -> str:
    """
    Clear all uploaded documents.

    Returns:
        str: Status message.
    """
    global agent, tool_index

    for pdf in DATA_DIR.glob("*.pdf"):
        pdf.unlink()

    # Clear storage
    storage_path = Path("storage")
    if storage_path.exists():
        shutil.rmtree(storage_path)

    agent = None
    tool_index = None
    return "All documents cleared."

def get_status() -> str:
    """
    Get current indexing status.

    Returns:
        str: Status message."""
    indexed = get_indexed_files()
    if indexed:
        return f"Indexed {len(indexed)} documents: {', '.join(indexed)}"
    return "No documents indexed. Please upload PDFs."

def respond(message: str, history: List[Tuple[str, str]]) -> str:
    """
    Handle chat messages.

    Args:
        message (str): User's message.
        history (List[Tuple[str, str]]): Chat history.
    Returns:
        str: Agent's response.
    """
    global agent

    if agent is None:
        return "Please upload documents first before asking questions."

    try:
        # Run async chat
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(chat(agent, message))
        finally:
            loop.close()
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Build the UI
with gr.Blocks(title="Agentic RAG Assistant") as app:
    gr.Markdown("# Agentic RAG Assistant")
    gr.Markdown("Upload PDF documents and ask questions about them.")

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="Upload PDFs (max 5)",
                file_count="multiple",
                file_types=[".pdf"],
            )
            upload_btn = gr.Button("Upload & Index", variant="primary")
            clear_btn = gr.Button("Clear All Documents", variant="secondary")
            status_text = gr.Textbox(
                label="Status",
                value=get_status(),
                interactive=False,
            )

    chatbot = gr.ChatInterface(
        fn=respond,
        title="",
        examples=[
            "What is MetaGPT?",
            "Summarize the SelfRAG paper",
            "Compare the approaches in loftq and longlora",
        ],
    )

    # Event handlers
    upload_btn.click(
        fn=upload_files,
        inputs=[file_upload],
        outputs=[status_text],
    )
    clear_btn.click(
        fn=clear_documents,
        outputs=[status_text],
    )

# Initialize on startup
if __name__ == "__main__":
    print("Initializing agent...")
    status = initialize_agent()
    print(status)
    app.launch()
