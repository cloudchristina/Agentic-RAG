from pathlib import Path
import json
from typing import List, Tuple, Optional
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.objects import ObjectIndex
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from helper import get_groq_api_key

# 1. Initialize settings
def init_settings():
    api_key = get_groq_api_key()
    Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 2. Define storage paths
STORAGE_DIR = Path("storage")
DATA_DIR = Path("data")
MANIFEST_FILE = STORAGE_DIR / "manifest.json"

# 3. Get doc name
def get_doc(pdf_path: str) -> str:
    """
    Load document from the given PDF path.
    Args:
        pdf_path (str): Path to the PDF document.
    Returns:
        str: Document name without extension.

    For example, if pdf_path is "documents/report.pdf", the .stem property will return "report"

    """
    return Path(pdf_path).stem

# 4. Build tools for a single document
def build_doc_tools(pdf_path: str) -> Tuple[list[QueryEngineTool], VectorStoreIndex, SummaryIndex]:
    """
    Build vector and summary tools for a single document.

    Args:
        pdf_path (str): Path to the PDF document.
    Returns:
        Tuple[list, object, object]: A tuple containing a list of QueryEngineTools,
        a VectorStoreIndex, and a SummaryIndex.

    """
    doc_name = get_doc(pdf_path)

    # 1. load doc and parse
    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    # 2. create index
    vector_index = VectorStoreIndex(nodes)
    summary_index = SummaryIndex(nodes)

    # 3. create query engine
    vector_engine = vector_index.as_query_engine(similarity_top_k=2)
    summary_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_aync=True,
    )

    # 4. create tools
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_engine,
        name=f"vector_{doc_name}",
        description=f"Useful for answering questions about the content of the document {doc_name}."
    )

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_engine,
        name=f"summary_{doc_name}",
        description=f"Useful for summarizing the content of the document {doc_name}."
    )

    return [vector_tool, summary_tool], vector_index, summary_index

# 5. Save doc index for caching
def save_doc_index(doc_name:str, vector_index:VectorStoreIndex, summary_index:SummaryIndex):
    """
    Persist a document's indices to disk.

    Args:
        doc_name (str): Name of the document.
        vector_index (VectorStoreIndex): Vector index to save.
        summary_index (SummaryIndex): Summary index to save.

    Returns: None

    """
    # 1. create storage dir
    doc_storage = STORAGE_DIR / doc_name
    vector_path = doc_storage / "vector"
    summary_path = doc_storage / "summary"

    # 2. persist vector index
    vector_index.storage_context.persist(persist_dir=str(vector_path))

    # 3. persist summary index
    summary_index.storage_context.persist(persist_dir=str(summary_path))

# 6. Load cached doc index
def load_doc_index(doc_name: str) -> Optional[Tuple[VectorStoreIndex, SummaryIndex]]:
    """
    Load a document's indices from disk if exist.

    Args:
        doc_name (str): Name of the document.
    Returns:
        Optional[Tuple[VectorStoreIndex, SummaryIndex]]: A tuple containing the vector and summary indices,
        or None if the indices do not exist.

    """

    # check if paths exist what paths
    doc_storage = STORAGE_DIR / doc_name
    vector_path = doc_storage / "vector"
    summary_path = doc_storage / "summary"

    if not vector_path.exists() or not summary_path.exists():
        return None

    # load vector & summary index
    try:
        vector_ctx = StorageContext.from_defaults(persist_dir=str(vector_path))
        vector_index = load_index_from_storage(vector_ctx)
        summary_ctx = StorageContext.from_defaults(persist_dir=str(summary_path))
        summary_index = load_index_from_storage(summary_ctx)
        return vector_index, summary_index
    except Exception as e:
        print(f"Error loading index for {doc_name}: {e}")
        return None

# 7. Load manifest - Single source of truth of indexed documents
def load_manifest() -> dict:
    """
    Load the manifest of indexed documents if it exists.

    Returns:
        dict: A dictionary containing the manifest of indexed documents.

    """
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r") as f:
            return json.load(f)
    return {"indexed_files": []}

# 8. Save manifest
def save_manifest(indexed_files: List[str]):
    """
    Save the manifest of indexed documents.

    Args:
        indexed_files (List[str]): List of indexed document names.

    Returns: None

    """
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        json.dump({"indexed_files": indexed_files}, f, indent=2)

# 9. Rebuild or load tools for a document
def rebuild_tools_for_document(pdf_path: str) -> List[QueryEngineTool]:
    """
    Build tools for a document, using cache if available.

    Args:
        pdf_path (str): Path to the PDF document.

    Returns:
        List[QueryEngineTool]: List of QueryEngineTools for the document."""
    # 1. get doc name
    doc_name = get_doc(pdf_path)

    # 2. try load indices
    cached = load_doc_index(doc_name)
    if cached is not None:
        # use cached indices
        vector_index, summary_index = cached

        # create query engines
        vector_engine = vector_index.as_query_engine(similarity_top_k=2)
        summary_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_aync=True,
        )

        # create tools
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_engine,
            name=f"vector_{doc_name}",
            description=f"Useful for answering questions about the content of the document {doc_name}."
        )

        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_engine,
            name=f"summary_{doc_name}",
            description=f"Useful for summarizing the content of the document {doc_name}."
        )

        return [vector_tool, summary_tool]

    # 3. if fail, build tools and save indices
    tools, vector_index, summary_index = build_doc_tools(pdf_path)
    save_doc_index(doc_name, vector_index, summary_index)
    return tools

# 10. Build tools for all documents
def build_all_doc_tools(pdf_paths: List[str]) -> List[QueryEngineTool]:
    """
    Build tools for all documents.

    Args:
        pdf_paths (List[str]): List of paths to PDF documents.

    Returns:
        List[QueryEngineTool]: List of all QueryEngineTools for the documents.
        all_tools = []
        indexed_files = []
    """

    all_tools = []
    indexed_files = []

    for pdf_path in pdf_paths:
        tools = rebuild_tools_for_document(pdf_path)

        # extend
        all_tools.extend(tools)

        # append doc name to indexed files
        indexed_files.append(get_doc(pdf_path))

    save_manifest(indexed_files)
    return all_tools

# 11. Create tool index for dynamic retrieval
def create_tool_index(tools: List[QueryEngineTool]) -> ObjectIndex:
    """
    Create an ObjectIndex over all tools for dynamic retrieval.

    Args:
        tools (List[QueryEngineTool]): List of QueryEngineTools.

    Returns:
        ObjectIndex: An ObjectIndex over the tools.
    """
    return ObjectIndex.from_objects(tools, index_cls=VectorStoreIndex)

# 12. Get tool retriever
def get_tool_retriever(tool_index: ObjectIndex, top_k: int = 2):
    """
    Get a retriever that returns top-k relevant tools.

    Args:
        tool_index (ObjectIndex): The tool index.
        top_k (int): Number of top relevant tools to retrieve.  Defaults to 2.

    Returns:
        Retriever: A retriever for the tool index.
    """
    return tool_index.as_retriever(similarity_top_k=top_k)

# 13. Get list of indexed files
def get_indexed_files() -> List[str]:
    """
    Return list of currently indexed document names.

    Returns:
        List[str]: List of indexed document names.
    """
    manifest = load_manifest()
    return manifest.get("indexed_files", [])
