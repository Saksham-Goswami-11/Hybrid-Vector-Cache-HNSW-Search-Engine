import asyncio
import os
import shutil
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ingest import IngestionDaemon

async def run_mcp_test():
    test_dir = "./knowledge_base_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    test_file = os.path.join(test_dir, "onboarding.md")
    
    print("Writing test document...")
    onboarding_content = """
# Office Working Hours

Our company onboarding rules:
Working hours are from 9 AM to 5 PM.
We expect everyone to join the standup daily at 9:30 AM.
The secret database key is 10-20-30.
    """
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(onboarding_content)

    print("Running Ingestion...")
    daemon = IngestionDaemon(directory=test_dir, port=6380)
    daemon.scan_directory()

    # Inject target directory to stdio subprocess
    env = os.environ.copy()
    env["KNOWLEDGE_BASE_DIR"] = test_dir
    env["NEARBY_PORT"] = "6380"
    
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server/server.py"],
        env=env
    )

    print("Connecting to FastMCP Server over stdio...")
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            print("Listing tools...")
            tools = await session.list_tools()
            print(f"Discovered Tools: {[t.name for t in tools.tools]}")
            assert any(t.name == "search_internal_knowledge" for t in tools.tools), "Tool not found!"

            print("Calling search_internal_knowledge tool...")
            result = await session.call_tool(
                "search_internal_knowledge",
                arguments={"query": "What time is standup and what is the secret database key?"}
            )
            
            content_text = result.content[0].text
            print(f"Tool Result content:\n{content_text}")
            
            assert "10-20-30" in content_text, "Expected secret database key in tool output!"
            assert "9:30 AM" in content_text, "Expected standup time in tool output!"
            assert "onboarding.md" in content_text, "Expected filename citation in tool output!"

    print("Cleaning up test directory...")
    shutil.rmtree(test_dir)
    print("MCP Server validation test passed!")

if __name__ == "__main__":
    asyncio.run(run_mcp_test())
