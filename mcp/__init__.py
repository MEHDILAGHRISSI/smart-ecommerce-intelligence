"""Package MCP — Model Context Protocol Server & Client."""
from .client import MCPClient, build_mcp_system_prompt
from .server import SmartEcommerceMCPServer

__all__ = ["MCPClient", "SmartEcommerceMCPServer", "build_mcp_system_prompt"]