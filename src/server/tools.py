# RAG tools for the realtime voice agent, built per-connection so
# knowledge_base_search is scoped to the caller.
from src.server.KB import KnowledgeBase

kb = KnowledgeBase()


def web_search(query: str) -> str:
    """Real-time web search via tavily."""
    try:
        text, imgs, links = kb.fetchContextWeb(query)
        # Keep the tool output small enough for the realtime model
        return (text or "No relevant web results found.")[:4000]
    except Exception as e:
        print("ERROR in web_search tool:", e)
        return "Web search failed."


def build_tools(user_id: str = "", tenant_id: str = ""):
    """Return the tool set for a voice connection, with KB search bound to the caller."""

    def knowledge_base_search(query: str, mode: str = "specific") -> str:
        """Collapsed-tree search over the RAPTOR knowledge base for this user.

        mode='broad' for summary/overview questions (covers all themes via summary nodes),
        mode='specific' for targeted questions (leaf-level detail).
        """
        try:
            text, links = kb.fetchContextDB(query, user_id, tenant_id, mode=mode)
            return text or "No relevant context found in the uploaded documents."
        except Exception as e:
            print("ERROR in knowledge_base_search tool:", e)
            return "Knowledge base search failed."

    return _tool_specs(knowledge_base_search)


def _tool_specs(knowledge_base_search):
    return [
        {
            "name": "knowledge_base_search",
            "description": "Searches the uploaded documents for this session and returns relevant context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query from the user."},
                    "mode": {
                        "type": "string",
                        "enum": ["specific", "broad"],
                        "description": "Use 'broad' for summary/overview/whole-document questions, 'specific' for a targeted question.",
                    },
                },
                "required": ["query"],
            },
            "func": knowledge_base_search,
        },
        {
            "name": "web_search",
            "description": "Searches the real-time web and provides more information about topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query from the user."},
                },
                "required": ["query"],
            },
            "func": web_search,
        },
    ]
