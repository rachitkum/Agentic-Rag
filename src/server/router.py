"""
Routing, query-rewriting, and retrieval-grading layer.

These are the decision primitives of the RAG agent. They are intentionally small,
stateless, and dependency-injected (client/model passed in) so the agentic retrieval
loop in LLM.py can call them repeatedly without duplicating logic.

Turn flow (see LLM.py agenticAnswer):
    route()          -> rewrite the message into a standalone query + pick route/mode
    [loop] retrieve -> grade_context() -> reformulate() if insufficient -> retrieve
    answer

The follow-up fix lives in route(): elliptical messages are rewritten into
self-contained queries BEFORE anything is embedded or searched.

    "what about its risks?"   (history: talking about the RAPTOR method)
        -> standalone_query = "What are the risks of the RAPTOR retrieval method?"
"""

from pydantic import BaseModel
from typing import Literal


# ---- route() output --------------------------------------------------------
class RouteDecision(BaseModel):
    # docs = uploaded KB, web = internet, chat = small talk, reformat = previous answer
    route: Literal["docs", "web", "chat", "reformat"]
    standalone_query: str
    mode: Literal["specific", "broad"]


# ---- grade_context() output ------------------------------------------------
class ContextGrade(BaseModel):
    # Is the retrieved context enough to fully answer the standalone query?
    sufficient: bool
    # If not sufficient, a sharper/alternate query to try next.
    next_query: str
    # If not sufficient, whether to escalate retrieval breadth or switch source.
    escalate_mode: Literal["specific", "broad"]
    switch_source: Literal["same", "docs", "web"]


_ROUTE_SYSTEM = """You are the routing and query-rewriting layer of a RAG agent.

You are given the recent conversation and the user's newest message. Do THREE things:

1) REWRITE the newest message into a standalone, self-contained search query.
   - Resolve all pronouns and references using the conversation history
     ("it", "that", "the second one", "explain more" -> name the actual subject).
   - If the message is already self-contained, keep it as-is.
   - The rewrite must make sense on its own with NO conversation context.

2) DECIDE the route:
   - "chat": greetings, small talk, thanks, "what can you do" — no retrieval.
   - "reformat": the user wants you to transform or re-present the PREVIOUS answer
     ("summarize what you just said", "put that in a table", "make it shorter",
     "translate that") — this needs NO new retrieval, only the last answer.
   - "docs": asking about their uploaded documents / study material, or continuing
     a thread that was answered from documents.
   - "web": wants current, real-world, or general internet information not expected
     to be in their uploaded documents.

3) DECIDE the retrieval mode (only meaningful for docs/web):
   - "broad": summaries, overviews, "the whole document", "everything about",
     anything needing coverage across all sections.
   - "specific": a targeted question about one fact, section, or detail.

Return ONLY the structured decision."""


_GRADE_SYSTEM = """You judge whether retrieved context is enough to fully and
accurately answer a query. Be strict: partial or tangential context is NOT
sufficient.

Given the query and the retrieved context, decide:
- sufficient: true only if the context clearly contains what's needed to answer.
- If NOT sufficient:
  - next_query: a sharper or alternate phrasing likely to retrieve the missing
    piece (not a repeat of the same query).
  - escalate_mode: "broad" if the question needs coverage across sections and the
    current context looks too narrow; otherwise "specific".
  - switch_source: "web" if the answer likely isn't in uploaded documents and needs
    the internet; "docs" to go back to uploaded documents; "same" to keep the
    current source.

If sufficient is true, next_query/escalate_mode/switch_source are ignored."""


def _history_block(history: list) -> str:
    if not history:
        return "(no prior conversation)"
    lines = []
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if isinstance(content, list):  # multimodal content -> take the text parts
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# activeButton values the client may send to FORCE a source.
_FORCE_WEB = "web search"
_FORCE_DOCS = "document"


def route(client, model: str, user_message: str, history: list, active_button: str = "") -> RouteDecision:
    """Rewrite the message into a standalone query and pick the route + mode."""
    messages = [
        {"role": "system", "content": _ROUTE_SYSTEM},
        {"role": "user", "content": (
            f"Conversation so far:\n{_history_block(history)}\n\n"
            f"Newest user message:\n{user_message}"
        )},
    ]

    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            max_tokens=250,
            temperature=0,
            response_format=RouteDecision,
        )
        decision = completion.choices[0].message.parsed
    except Exception as e:
        # Fail safe: never block the turn on a routing error.
        print("ERROR in router, falling back to raw query:", e)
        decision = RouteDecision(route="docs", standalone_query=user_message, mode="specific")

    # An explicit UI toggle wins over the LLM's source choice, but keeps the
    # rewritten query and mode. Never overrides chat/reformat.
    if decision.route in ("docs", "web"):
        if active_button == _FORCE_WEB:
            decision.route = "web"
        elif active_button == _FORCE_DOCS:
            decision.route = "docs"

    print(f"ROUTE -> {decision.route} | mode={decision.mode} | q='{decision.standalone_query}'")
    return decision


def grade_context(client, model: str, query: str, context: str) -> ContextGrade:
    """Decide whether `context` is enough to answer `query`; if not, say what to try
    next. This is what makes the agent 'know' when it needs to fetch more."""
    # No context at all -> definitely insufficient; try a broader pass.
    if not context or not context.strip():
        return ContextGrade(sufficient=False, next_query=query, escalate_mode="broad", switch_source="same")

    messages = [
        {"role": "system", "content": _GRADE_SYSTEM},
        {"role": "user", "content": f"Query:\n{query}\n\nRetrieved context:\n{context}"},
    ]
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            max_tokens=200,
            temperature=0,
            response_format=ContextGrade,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        # Fail safe: treat what we have as sufficient rather than looping blindly.
        print("ERROR in grade_context, treating context as sufficient:", e)
        return ContextGrade(sufficient=True, next_query=query, escalate_mode="specific", switch_source="same")
