"""Package MCP Server — Model Context Protocol Server & Client."""
from .client import MCPClient, build_mcp_system_prompt
from .server import SmartEcommerceMCPServer, build_server, run_stdio

__all__ = ["MCPClient", "SmartEcommerceMCPServer", "build_server", "run_stdio", "build_mcp_system_prompt"]
