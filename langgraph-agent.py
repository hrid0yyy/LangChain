from typing import Annotated

from langchain_core.messages import BaseMessage  # Message type for conversation
from langchain_core.tools import tool, Tool      # Tool decorators and base class
from langchain_community.tools import DuckDuckGoSearchResults  # DuckDuckGo search tool
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END  # Graph building blocks
from langgraph.graph.message import add_messages    # Message handling for state
from langgraph.prebuilt import ToolNode, tools_condition  # Tool node and condition helpers

from dotenv import load_dotenv  # For loading environment variables
from langchain_google_genai import ChatGoogleGenerativeAI  # Google Gemini LLM

from langgraph.checkpoint.memory import MemorySaver  # In-memory checkpointing

# Initialize in-memory checkpointing for conversation state
memory = MemorySaver()

# Load environment variables (API keys, etc.)
load_dotenv()

# Initialize the Google Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Define the state structure for the graph (holds message history)
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create a state graph for the agent
graph_builder = StateGraph(State)

# Instantiate the DuckDuckGo search tool
ddg_tool = DuckDuckGoSearchResults()

# Wrapper function to ask user permission before using DuckDuckGo
def ask_duckduckgo_permission(query: str) -> str:
    print(f"\n[Permission Needed] The assistant wants to use DuckDuckGo to answer: '{query}'")
    consent = input("Allow DuckDuckGo search? (y/n): ").strip().lower()
    if consent == "y":
        # If user consents, perform the search
        return ddg_tool.invoke({"query": query})
    else:
        # If user denies, return a message
        print("[DuckDuckGo search denied by user.]")
        return "User denied DuckDuckGo search."

# Register the permission-wrapped DuckDuckGo tool
duckduckgo_permission_tool = Tool.from_function(
    name="duckduckgo_search_with_permission",
    description="Search DuckDuckGo with user permission.",
    func=ask_duckduckgo_permission,
)

# List of tools available to the agent (start with DuckDuckGo)
tools = [duckduckgo_permission_tool]

# Bind the tools to the LLM so it can call them
llm_with_tools = llm.bind_tools(tools)

# The main chatbot node: generates a response using the LLM and tools
def chatbot(state: State):
    # LLM generates a response based on the conversation history
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Simulate human-in-the-loop: prompt a human for input when needed
def interrupt(data):
    print(f"\n[Human Assistance Needed] LLM requested help with: {data['query']}")
    human_input = input("[Human] Please provide your answer: ")
    return {"data": human_input}

# Tool for requesting human assistance (calls interrupt)
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

# Register the human assistance tool
human_assistance_tool = Tool.from_function(
    name="human_assistance",
    description="Request assistance from a human.",
    func=human_assistance,
)

# Add the human assistance tool to the tools list
tools.append(human_assistance_tool)

# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Add the tool node (with all tools) to the graph
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Add conditional edges: decide when to use tools
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile the graph with memory checkpointing
graph = graph_builder.compile(checkpointer=memory)

# Function to run the chatbot with message history and thread id
def run_chatbot(messages: list[BaseMessage], thread_id: str = "1") -> list[BaseMessage]:
    state = {"messages": messages}
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(state, config)
    return result["messages"]

# Main interactive chat loop
if __name__ == "__main__":
    messages = []  # Conversation history
    thread_id = "1"
    print("Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        # Get assistant response
        response = run_chatbot(messages, thread_id=thread_id)
        assistant_message = response[-1]
        print(f"Assistant: {assistant_message.content}")
        # Add assistant message to history
        messages.append({"role": "assistant", "content": assistant_message.content})
