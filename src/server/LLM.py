from openai import AzureOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import os

from src.server import router

# Load environment variables from the .env file (if present)
load_dotenv()

AZURE_API_KEY = os.getenv('AZURE_API_KEY')
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION')
AZURE_API_ENDPOINT = os.getenv('AZURE_API_ENDPOINT')

AZURE_CLIENT = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_API_ENDPOINT,
    api_version=AZURE_API_VERSION,
)


class UnderstandResponse(BaseModel):
    definition: str
    detailed_explanation: str
    analogies_and_examples: str
    suggested_questions: List[str]


class ChatOpenAI:
    def __init__(self, model="gpt-4o-mini") -> None:
        self.client = AZURE_CLIENT
        self.model = model

    async def simpleResponse(self, msg):
        # Message contains system msg, chat history and user current query
        completion = self.client.chat.completions.create(model=self.model, messages=msg, max_tokens=5000, temperature=0.1)
        return completion.choices[0].message

    # Summarize a cluster of chunks into one node (used when building the RAPTOR tree)
    def summarizeCluster(self, texts: list) -> str:
        joined = "\n\n---\n\n".join(texts)
        messages = [
            {"role": "system", "content": "You compress a set of related passages into one dense, faithful summary that preserves every distinct fact, topic and detail. Do not add information. Do not editorialize."},
            {"role": "user", "content": f"Summarize the following passages into a single coherent summary that keeps all key points:\n\n{joined}"},
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=800,
            temperature=0.1,
        )
        return completion.choices[0].message.content

    # Max retrieve -> grade -> reformulate cycles before we answer with what we have.
    MAX_RETRIEVAL_ITERS = 2

    # Casual / small-talk reply.
    def _casualReply(self, standalone_query, history):
        hist = ' '.join(str(d.get('content', '')) + ", " for d in history)
        messages = [
            {"role": "system", "content": "You are rag-agent, a friendly document assistant. Your role here is to engage in light, informal chats — no document or technical topics in this mode. Respond to greetings and light questions (e.g., 'Hi', 'Hello', 'What can you do for me?') using warm, friendly, and informal language that feels natural and welcoming."},
            {"role": "user", "content": "\nhistorical message: " + hist + "\n\ncurrent message: " + standalone_query},
        ]
        casual = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=60, temperature=0
        )
        return casual.choices[0].message

    # "reformat" route: transform the previous answer with no new retrieval.
    def _reformatReply(self, standalone_query, history):
        last_assistant = ""
        for turn in reversed(history):
            if turn.get("role") == "assistant":
                content = turn.get("content", "")
                last_assistant = content if isinstance(content, str) else str(content)
                break
        messages = [
            {"role": "system", "content": "You transform your OWN previous answer as the user asks (summarize, shorten, tabulate, translate, re-explain). Work only from the previous answer below — do not invent new facts."},
            {"role": "user", "content": f"Previous answer:\n{last_assistant}\n\nUser request:\n{standalone_query}"},
        ]
        completion = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=1200, temperature=0.2
        )
        return completion.choices[0].message

    # Grounded, structured answer. Uses the rewritten standalone query (not the raw
    # pronoun message) so generation and retrieval stay consistent on follow-ups.
    def _groundedAnswer(self, sysMsg, history, standalone_query, context):
        msg = [sysMsg] + history + [{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {standalone_query}",
        }]
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=msg,
            max_tokens=1500,
            temperature=0.1,
            response_format=UnderstandResponse,
        )
        return completion.choices[0].message

    # How many strong (high-confidence) chunks we consider "enough" to answer well.
    ENOUGH_STRONG_CHUNKS = 3

    # Confidence-based agentic retrieval over the uploaded documents.
    #
    # 1. Retrieve + score each chunk with the cross-encoder.
    # 2. Split strong vs. weak by KB.STRONG_THRESHOLD.
    # 3. If enough strong chunks -> stop, answer from them (fast path).
    # 4. Else -> the WEAK chunks failed; reformulate a sharper query and re-retrieve
    #    to REPLACE them. Keep the strong chunks we already have. Max MAX_RETRIEVAL_ITERS.
    # 5. Merge all strong chunks collected across iterations and generate ONCE
    #    (no parallel/partial answer -> no contradiction/flip-flop to reconcile).
    def _agenticRetrieveDocs(self, kb, query, mode, user_id, tenant_id):
        collected_strong = []       # accumulates strong chunks across iterations
        seen_text = set()
        best_links = []

        for i in range(self.MAX_RETRIEVAL_ITERS):
            scored = kb.fetchScoredContext(query, user_id, tenant_id, mode=mode)

            for c in scored["strong"]:
                if c["text"] not in seen_text:
                    seen_text.add(c["text"])
                    collected_strong.append(c)
                    if c.get("link"):
                        best_links.append(c["link"])

            # Enough high-confidence context, or last allowed pass -> stop.
            if len(collected_strong) >= self.ENOUGH_STRONG_CHUNKS:
                print(f"AGENT: enough strong chunks ({len(collected_strong)}) after iter {i + 1}")
                break
            if i == self.MAX_RETRIEVAL_ITERS - 1:
                break

            # Weak chunks dominated -> reformulate around what's still missing and
            # escalate breadth, then retry to replace them.
            weak_preview = " | ".join(c["text"][:200] for c in scored["weak"][:3])
            grade = router.grade_context(self.client, self.model, query, weak_preview)
            print(f"AGENT: only {len(collected_strong)} strong after iter {i + 1}, re-retrieving")
            query = grade.next_query or query
            mode = grade.escalate_mode

        # Merge strong chunks; if we never found any strong ones, fall back to the
        # best-scored chunks we did see so we still attempt an answer.
        if not collected_strong:
            fallback = kb.fetchScoredContext(query, user_id, tenant_id, mode=mode)["all"][:3]
            collected_strong = fallback
            best_links = [c["link"] for c in fallback if c.get("link")]

        merged_text = "\n\n".join(c["text"] for c in collected_strong)
        return merged_text, [], best_links

    # Web retrieval keeps the single-blob Tavily path (not chunk-scored), still bounded.
    def _agenticRetrieveWeb(self, kb, query, mode, user_id, tenant_id):
        best_context, best_imgs, best_links = "", [], []
        for i in range(self.MAX_RETRIEVAL_ITERS):
            context, imgs, links = kb.fetchContext(query, "web search", user_id=user_id, tenant_id=tenant_id, mode=mode)
            if len(context) > len(best_context):
                best_context, best_imgs, best_links = context, imgs, links
            if i == self.MAX_RETRIEVAL_ITERS - 1:
                break
            grade = router.grade_context(self.client, self.model, query, context)
            if grade.sufficient:
                break
            query = grade.next_query or query
            mode = grade.escalate_mode
        return best_context, best_imgs, best_links

    # RAG chat entrypoint.
    #
    # ROUTE (rewrite the message into a standalone query using history + pick
    # route/mode) -> handle chat / reformat / retrieval. Retrieval AND generation both
    # use the rewritten standalone_query, which is what makes follow-ups ("what about
    # its risks?") work end to end. For retrieval routes the agent self-decides whether
    # it has enough context and loops if not (_agenticRetrieve).
    def simpleResponseWithToolCall(self, msg, kb, activeButton, query, history, user_id="", tenant_id=""):
        sysMsg = msg[0]  # system message assembled by the caller

        decision = router.route(
            client=self.client,
            model=self.model,
            user_message=query,
            history=history,
            active_button=activeButton,
        )

        # Small talk — no retrieval.
        if decision.route == "chat":
            return self._casualReply(decision.standalone_query, history), [], []

        # Operate on the previous answer — no retrieval.
        if decision.route == "reformat":
            return self._reformatReply(decision.standalone_query, history), [], []

        # Retrieval route. Docs uses confidence-based scoring + re-retrieval of weak
        # chunks; web uses the bounded blob loop.
        if decision.route == "web":
            context, imgs, relevant_link = self._agenticRetrieveWeb(
                kb, decision.standalone_query, decision.mode, user_id, tenant_id
            )
        else:
            context, imgs, relevant_link = self._agenticRetrieveDocs(
                kb, decision.standalone_query, decision.mode, user_id, tenant_id
            )

        answer = self._groundedAnswer(sysMsg, history, decision.standalone_query, context)
        return answer, imgs, relevant_link
