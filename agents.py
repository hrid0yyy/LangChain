from GroqCloud import llama
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

llm = llama()
search = DuckDuckGoSearchResults()

prompt = hub.pull("hwchase17/react")

# Do the reasoning with the LLM and the search tool
agent = create_react_agent(llm=llm, tools=[search], prompt=prompt)

# Create an agent executor that will run the agent with the tools
agent_executor = AgentExecutor(agent=agent, tools=[search], verbose=True)

response = agent_executor.invoke(
    {
        "input": "3 ways to reach dhaka from noakhali?",
    }
)

print(response)