import asyncio
from langchain.agents.middleware import SummarizationMiddleware
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
import os
from dotenv import load_dotenv

if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]
# Reload from .env with override=True to ensure fresh load
load_dotenv(override=True)

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# System prompt for the Service Desk Agent
SYSTEM_PROMPT = """
        You are a Linux applications deployment assistant operating via MCP tools.
        
        RULES (NON-NEGOTIABLE):
        1. Always fetch the get_app_deployment_configuration_tool to understand supported applications before starting to answer.
        2. Never execute deployments unless explicitly requested by the user.
        3. List out this plan to user and always follow this order for if user asks for application deployment:
           - checkout_repository
           - build_application
           - verify_artifact
           - deploy_artifact
           - restart_application
           - get_application_status
        4. Additionally, support get_recent_logs_tool, get_all_services_status_on_server_tool and get_server_health_summary_tool tools
        5. Never assume success.
        6. Never suggest commands outside available MCP tools.
        7. Never fabricate command output.
        8. Prefer safety over speed.
        
        If unsure, ask for clarification.
        """

async def initialize_agent():
    """Initialize the agent with MCP tools."""
    # Initialize the LLM
    model = init_chat_model("gpt-4.1")
    famvest_api_key = os.getenv('FAMVEST_MCP_SERVER_API_KEY')
    # Initialize the MCP client with service desk configuration
    client = MultiServerMCPClient(
        {
            "linux-app-deployer": {
                "transport": "streamable_http",
                "url": "https://mcp.adminhub.famvest.online/mcp",
                "headers": {
                    "X-API-Key": famvest_api_key
                },
            },
            "famvest-app": {
                "transport": "streamable_http",
                "url": "https://mcp.trade.famvest.online/mcp",

            }
        }
    )
    tavily_search_tool = TavilySearch(max_results=5, topic="general")
    mcp_tools = await client.get_tools()
    tools = mcp_tools + [tavily_search_tool]

    # Create the agent with in-memory checkpointing
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        middleware=[SummarizationMiddleware(
            model="gpt-5-nano",
            trigger=("tokens", 500),  # Trigger summarization after every 550 tokens
            keep=("tokens", 200),  # Keep the last 200 tokens in full
        )],
    )
    return agent

async def main():
    """Main interactive loop."""
    agent = await initialize_agent()
    config = {"configurable": {"thread_id": "thread_1"}}

    while True:
        question = input("Enter your question (or 'bye' to quit): ")
        if question.strip().lower() == "bye":
            print("Bye bye! Tschüss!")
            break
        try:
            print("-" * 80)
            print("💡 ANSWER:")
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=question)]},
                config=config
            )
            print(result["messages"][-1].content)
            print("-" * 80)
        except Exception as e:
            print(f"❌ Error: {e}")

# Interactive loop
if __name__ == "__main__":
    asyncio.run(main())

