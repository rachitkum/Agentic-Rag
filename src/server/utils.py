import asyncio
import aiohttp
from typing import AsyncIterator, TypeVar
from starlette.websockets import WebSocket
from dotenv import load_dotenv
import os
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer

# Load environment variables from the .env file (if present)
load_dotenv()

TRANSCRIPTION_URL = os.getenv('TRANSCRIPTION_URL')

T = TypeVar("T")

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def getEmbeddings(query):
    try:
        embeddings = model.encode(query)
        return embeddings
    except Exception as e:
        print("ERROR ocurred while transforming query to embeddings: ", e)


def re_rank_cross_encoders(prompt: str, documents: list[str], linkResult: list[str], top_k: int = 3) -> tuple[str, list[int], list[str]]:
    """Re-ranks documents using a cross-encoder model for more accurate relevance scoring.

    Uses the MS MARCO MiniLM cross-encoder model to re-rank the input documents based on
    their relevance to the query prompt. Returns the concatenated text of the top_k most
    relevant documents along with their indices.

    Args:
        documents: List of document strings to be re-ranked.
        top_k: How many top documents to keep (broad/summary queries pass a higher value).

    Returns:
        tuple: A tuple containing:
            - relevant_text (str): Concatenated text from the top ranked documents
            - relevant_text_ids (list[int]): List of indices for the top ranked documents
            - relevant_link (list[str]): Source links for the top ranked documents

    Raises:
        ValueError: If documents list is empty
        RuntimeError: If cross-encoder model fails to load or rank documents
    """
    try:
        relevant_text = ""
        relevant_text_ids = []
        relevant_link = []
        if not documents:
            return "", [], []
        top_k = min(top_k, len(documents))
        ranks = encoder_model.rank(prompt, documents, top_k=top_k)
        for rank in ranks:
            if rank["score"] > 0.4:
                relevant_text += documents[rank["corpus_id"]]
                relevant_text_ids.append(rank["corpus_id"])
                relevant_link.append(linkResult[rank["corpus_id"]])

        print(relevant_link)
        return relevant_text, relevant_text_ids, relevant_link
    except Exception as e:
        print("ERROR ocurred while re-ranking context: ", e)
        return "", [], []


def rerank_scored(prompt: str, documents: list[str], linkResult: list[str], top_k: int = 8) -> list[dict]:
    """Re-rank and return per-chunk records with their cross-encoder scores.

    Unlike re_rank_cross_encoders (which concatenates and drops scores), this keeps
    each chunk's score so callers can split strong vs. weak by a confidence threshold.

    Returns a list of {"text": str, "score": float, "link": dict} sorted best-first.
    """
    try:
        if not documents:
            return []
        top_k = min(top_k, len(documents))
        ranks = encoder_model.rank(prompt, documents, top_k=top_k)
        out = []
        for rank in ranks:
            cid = rank["corpus_id"]
            out.append({
                "text": documents[cid],
                "score": float(rank["score"]),
                "link": linkResult[cid] if cid < len(linkResult) else None,
            })
        return out
    except Exception as e:
        print("ERROR ocurred while re-ranking (scored): ", e)
        return []


# Real-time calling socket: yields raw text frames from the browser
async def websocket_stream(websocket: WebSocket) -> AsyncIterator[str]:
    while True:
        data = await websocket.receive_text()
        yield data


# Merge multiple async streams into one stream (used by the realtime voice agent)
async def amerge(**streams: AsyncIterator[T]) -> AsyncIterator[tuple[str, T]]:
    """Merge multiple streams into one stream."""
    nexts: dict[asyncio.Task, str] = {
        asyncio.create_task(anext(stream)): key for key, stream in streams.items()
    }
    while nexts:
        done, _ = await asyncio.wait(nexts, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            key = nexts.pop(task)
            stream = streams[key]
            try:
                yield key, task.result()
                nexts[asyncio.create_task(anext(stream))] = key
            except StopAsyncIteration:
                pass
            except Exception as e:
                for task in nexts:
                    task.cancel()
                raise e


# TODO: Change it with deepgram STT
async def speech2text(path):
    data = {
        "url": path,
    }
    try:
        transcription_url = TRANSCRIPTION_URL

        async with aiohttp.ClientSession() as session:
            async with session.post(
                transcription_url,
                json=data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    print(f"Error: API responded with status {response.status}")
                    return ""

                response_data = await response.json()
                transcription = response_data.get("transcription", "")

                return transcription

    except aiohttp.ClientError as error:
        print("Network Error while speech-to-text:", error)
        return ""

    except Exception as error:
        print("Error while speech-to-text:", error)
        return ""
