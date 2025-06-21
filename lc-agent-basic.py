import os
import getpass
from langchain_core.messages import AnyMessage
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

checkpointer = InMemorySaver()

def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

model = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0
)

def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:  
    user_name = config["configurable"].get("user_name")
    system_msg = f"You are a helpful assistant. Address the user as {user_name}."
    return [{"role": "system", "content": system_msg}] + state["messages"]

class WeatherResponse(BaseModel):
    conditions: str

agent = create_react_agent(
    model=model,
    tools=[get_weather],
    prompt=prompt,
    checkpointer=checkpointer,
    response_format=WeatherResponse  
)

# Run the agent
config = {"configurable": {"user_name": "Neo Anderson", "thread_id": "1"}}
ask = "What is the weather like in Chikmagalur?"
sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": ask}]},
    config  
)
print(sf_response["structured_response"])

print("from ai - ", model.invoke(ask).content)