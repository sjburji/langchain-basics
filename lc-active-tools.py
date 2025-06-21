import getpass
import os
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

if not os.environ.get("TAVILY_API_KEY"):
  os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter API key for Tavily: ")

search = TavilySearch(max_results=2)
tools = [search]

model = init_chat_model("gpt-4o-mini", model_provider="openai")

query = "Hi!"
response = model.invoke([{"role": "user", "content": query}])
print(response.text())

agent_executor = create_react_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

input_message = {"role": "user", "content": "Hi, I'm Bob!"}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()

input_message = {"role": "user", "content": "What's my name?"}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()

input_message = {"role": "user", "content": "Search for the weather in SF"}
for step, metadata in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="messages"
):
    if metadata["langgraph_node"] == "agent" and (text := step.text()):
        print(text, end="|")