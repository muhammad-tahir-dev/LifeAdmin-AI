"""Plain-Python entry point for Member 3's CrewAI agents.

CrewAI wrapping (decorator import path differs between CrewAI versions):

    from crewai.tools import tool      # or: from crewai_tools import tool
    from research.tool import research_for_agent

    @tool("Research official information")
    def research_tool(query: str) -> str:
        \"\"\"Find verified, sourced information for a user's real-life task.\"\"\"
        return research_for_agent(query)
"""

from .pipeline import run_research
from .runtime import get_runtime


def research_for_agent(query, user_id=None, force_web=False):
    """Run the pipeline and return one compact text block for an agent prompt."""
    runtime = get_runtime()
    result = run_research(
        query,
        collection=runtime.collection,
        tavily_client=runtime.tavily_client,
        user_id=user_id,
        force_web=force_web,
    )

    lines = [result["answer"], "", f"Status: {result['status']}"]
    if result["conflicts"]:
        lines.append("Conflicts: " + " | ".join(result["conflicts"]))
    if result["sources"]:
        lines.append("Sources:")
        for source in result["sources"]:
            lines.append(
                f"- [{source['source_type']}] {source['title']} "
                f"{source['url']} (checked {source.get('retrieved_at') or 'n/a'})"
            )
    return "\n".join(lines)
