from openai import AzureOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import os

from src.server import router

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
        completion = self.client.chat.completions.create(model=self.model, messages=msg, max_tokens=5000, temperature=0.1)
        return completion.choices[0].message

    # One node from a cluster of chunks, used when building the RAPTOR tree.
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

    # Max retrieve -> grade -> reformulate cycles before answering with what we have.
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

    # Uses the rewritten standalone query so generation and retrieval stay consistent.
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

    # Strong chunks considered "enough" to answer.
    ENOUGH_STRONG_CHUNKS = 3

    # Retrieve -> score -> if not enough strong chunks, reformulate and retry to
    # replace the weak ones. Merge across iterations and generate once.
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

            # Enough context, or last allowed pass.
            if len(collected_strong) >= self.ENOUGH_STRONG_CHUNKS:
                print(f"AGENT: enough strong chunks ({len(collected_strong)}) after iter {i + 1}")
                break
            if i == self.MAX_RETRIEVAL_ITERS - 1:
                break

            # Weak chunks dominated -> reformulate and escalate breadth.
            weak_preview = " | ".join(c["text"][:200] for c in scored["weak"][:3])
            grade = router.grade_context(self.client, self.model, query, weak_preview)
            print(f"AGENT: only {len(collected_strong)} strong after iter {i + 1}, re-retrieving")
            query = grade.next_query or query
            mode = grade.escalate_mode

        # No strong chunks at all -> fall back to the best we saw.
        if not collected_strong:
            fallback = kb.fetchScoredContext(query, user_id, tenant_id, mode=mode)["all"][:3]
            collected_strong = fallback
            best_links = [c["link"] for c in fallback if c.get("link")]

        merged_text = "\n\n".join(c["text"] for c in collected_strong)
        return merged_text, [], best_links

    # Single-blob Tavily path, not chunk-scored, still bounded.
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

    # Route (rewrite into a standalone query + pick route/mode) -> chat / reformat /
    # retrieval. The rewritten query is what makes follow-ups work end to end.
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

        # Docs uses confidence scoring + re-retrieval; web uses the bounded blob loop.
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
