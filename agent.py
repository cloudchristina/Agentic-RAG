from llama_index.core.agent import FunctionAgent
from llama_index.llms.groq import Groq

from helper import get_groq_api_key

def create_agent(tool_retriever):
    """
    Create a FunctionAgent with tool retriever.

    Args:
        tool_retriever: A retriever that provides relevant tools.
    Returns:
        FunctionAgent: An agent configured with the tool retriever and LLM.
    """
    api_key = get_groq_api_key()  # Ensure env loaded
    llm = Groq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=api_key)

    agent = FunctionAgent(
        tool_retriever=tool_retriever,
        llm=llm,
        verbose=True,
        system_prompt=(
            "You are a helpful research assistant that answers questions about documents. "
            "Use the available tools to search and summarize documents to answer user queries. "
            "Always cite which document you found the information in."
        ),
    )

    return agent

async def chat(agent: FunctionAgent, message: str) -> str:
    """
    Process a chat message and return the response.

    Args:
        agent (FunctionAgent): The agent to process the message.
        message (str): The user's message.
    Returns:
        str: The agent's response with tool usage info.
    """
    response = await agent.run(message)

    # Extract tool calls from response
    tool_info = []
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            tool_name = tc.tool_name if hasattr(tc, 'tool_name') else str(tc)
            tool_info.append(tool_name)

    result = str(response)
    if tool_info:
        result += f"\n\n---\n**Tools used:** {', '.join(tool_info)}"

    return result
