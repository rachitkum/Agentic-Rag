import uuid
import asyncio
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocket
from starlette.requests import Request

from src.server.utils import websocket_stream, speech2text
from src.server.LLM import ChatOpenAI
from src.server.KB import KnowledgeBase
from src.server.prompt import INSTRUCTIONS
from src.server.tools import build_tools
from src.server.voice import OpenAIVoiceReactAgent
from src.server.ingest import ingest_pdf
from src.db import auth, memory, postgres, vectorstore

DEFAULT_SYSTEM_PROMPT = (
    "You are rag-agent, a helpful study and research assistant. "
    "Answer using the retrieved context when available and be honest when you don't know."
)

# In-memory upload job tracking (job_id -> {status, result/error}).
# Fine for a single-process demo; swap for Redis if you scale out.
UPLOAD_JOBS: dict[str, dict] = {}


# Upload a PDF into a chat session. Chunk + RAPTOR tree build + Weaviate store runs
# in the background; the client polls /upload/status/{job_id}.
async def handleUpload(request: Request):
    try:
        form = await request.form()
        upload = form.get("file")
        user_id = form.get("user_id")
        session_id = form.get("session_id") or ""

        if upload is None:
            return JSONResponse({"err": "No file provided"}, status_code=400)
        if not user_id:
            return JSONResponse({"err": "user_id is required"}, status_code=400)

        tenant_id = await asyncio.to_thread(auth.resolveTenant, user_id)
        if tenant_id is None:
            return JSONResponse({"err": "Unknown user_id"}, status_code=404)

        pdf_bytes = await upload.read()
        file_name = getattr(upload, "filename", "") or "document.pdf"

        doc_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        UPLOAD_JOBS[job_id] = {"status": "processing"}

        async def run_job():
            try:
                # Tree-building calls the LLM per cluster; keep it off the event loop.
                result = await asyncio.to_thread(
                    ingest_pdf, pdf_bytes, user_id, tenant_id, doc_id, session_id, file_name
                )
                UPLOAD_JOBS[job_id] = {"status": "done", "result": result}
            except Exception as e:
                print("ERROR during ingestion job:", e)
                UPLOAD_JOBS[job_id] = {"status": "error", "error": str(e)}

        asyncio.create_task(run_job())

        return JSONResponse({"job_id": job_id, "doc_id": doc_id, "status": "processing"})
    except Exception as e:
        print("ERROR occurred while handling upload ", e)
        return JSONResponse({"err": "Internal server error"}, status_code=500)


# Poll the status of an ingestion job.
async def handleUploadStatus(request: Request):
    job_id = request.path_params["job_id"]
    job = UPLOAD_JOBS.get(job_id)
    if job is None:
        return JSONResponse({"err": "Unknown job_id"}, status_code=404)
    return JSONResponse(job)


# RAG chat endpoint.
# Request contains: user_id, session_id, user_content, and optional: system_content,
# activeButton, image, audio. History is read from Valkey, not sent by the client.
async def handleChat(request: Request):
    try:
        userMsg = await request.json()

        user_id = userMsg.get("user_id", "")
        if not user_id:
            return JSONResponse({"err": "user_id is required"}, status_code=400)

        tenant_id = await asyncio.to_thread(auth.resolveTenant, user_id)
        if tenant_id is None:
            return JSONResponse({"err": "Unknown user_id"}, status_code=404)

        session_id = userMsg.get("session_id", "")
        if not session_id:
            return JSONResponse({"err": "session_id is required"}, status_code=400)

        kb = KnowledgeBase()

        history = await asyncio.to_thread(memory.getHistory, user_id, session_id)

        sysMsg = {
            "role": "system",
            "content": userMsg.get("system_content") or DEFAULT_SYSTEM_PROMPT
        }

        currMsg = {
            "role": "user",
            "content": "Query: " + userMsg["user_content"]
        }

        # Helps to route request to chat bot
        routingCurrMsg = userMsg["user_content"]

        # TODO: Dont use image inputs with tool calling agents
        if "image" in userMsg and userMsg["image"] != "":
            content = [{"type": "text", "text": currMsg["content"]},
                       {"type": "image_url", "image_url": {"url": userMsg["image"]}}]
            currMsg["content"] = content

        elif "audio" in userMsg and userMsg["audio"] != "":
            transcription = await speech2text(userMsg["audio"])

            if transcription == "":
                return JSONResponse({"err": "Failed to process audio"})

            currMsg["content"] = str(transcription) + currMsg["content"]

        msg = [sysMsg] + history + [currMsg]
        llm = ChatOpenAI()

        shortHistory = history[-2:] if len(history) >= 2 else history
        gptResponse, web_img, revelant_link = llm.simpleResponseWithToolCall(
            msg=msg,
            kb=kb,
            activeButton=userMsg.get("activeButton", "document"),
            query=routingCurrMsg,
            history=shortHistory,
            user_id=user_id,
            tenant_id=tenant_id,
        )

        assistant = {
            "role": "assistant",
            "content": gptResponse.content,
        }

        if len(web_img) > 0:
            assistant["image"] = web_img

        if len(revelant_link) > 0:
            assistant["links"] = revelant_link

        # Store the plain text of both turns; images and links stay out of the prompt.
        await asyncio.to_thread(memory.appendTurns, user_id, session_id, [
            {"role": "user", "content": routingCurrMsg},
            {"role": "assistant", "content": gptResponse.content},
        ])

        return JSONResponse({"data": [assistant]})
    except Exception as e:
        print("ERROR occurred while processing chat ", e)
        return JSONResponse({"err": "Internal server error"})


# Realtime speech-to-speech call backed by the RAG tools.
# user_id comes in as a query param (ws://.../call?user_id=...) so the voice
# knowledge_base_search is scoped to that user's documents.
async def handleCall(websocket: WebSocket):
    await websocket.accept()

    user_id = websocket.query_params.get("user_id", "")
    tenant_id = await asyncio.to_thread(auth.resolveTenant, user_id) if user_id else None
    if tenant_id is None:
        await websocket.close(code=4004)
        return

    browser_receive_stream = websocket_stream(websocket)

    agent = OpenAIVoiceReactAgent(
        model="gpt-4o-realtime-preview",
        tools=build_tools(user_id, tenant_id),
        instructions=INSTRUCTIONS,
    )

    await agent.aconnect(browser_receive_stream, websocket.send_text)

# Create-or-login. - not added password auth for now 
async def createuser(request: Request):
    try:
        body = await request.json()
        user_id = (body.get("user_id") or "").strip()

        if not user_id:
            return JSONResponse({"err": "user_id is required"}, status_code=400)

        user = await asyncio.to_thread(auth.createOrLoginUser, user_id)

        return JSONResponse({
            "user_id": user["user_id"],
            "tenant_id": user["tenant_id"],
            "created": user["created"],
        }, status_code=201 if user["created"] else 200)
    except Exception as e:
        print("ERROR occurred while creating user ", e)
        return JSONResponse({"err": "Internal server error"}, status_code=500)


routes = [
    Route("/user",createuser,methods=["POST"]),
    Route("/upload", handleUpload, methods=["POST"]),
    Route("/upload/status/{job_id}", handleUploadStatus, methods=["GET"]),
    Route("/chat", handleChat, methods=["POST"]),
    WebSocketRoute("/call", handleCall),
]

async def on_startup():
    """Create the partitioned tables if they don't exist yet."""
    await asyncio.to_thread(postgres.initSchema)


async def on_shutdown():
    await asyncio.to_thread(vectorstore.close_client)
    await asyncio.to_thread(memory.close_client)


app = Starlette(debug=True, routes=routes,
                on_startup=[on_startup], on_shutdown=[on_shutdown])

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
