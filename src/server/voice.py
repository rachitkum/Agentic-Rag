# Minimal realtime speech-to-speech agent over the OpenAI Realtime API.
# Bridges a browser websocket stream with the model websocket and executes
# RAG tools (see tools.py) when the model requests a function call.
import json
import asyncio
import os
import aiohttp
from dotenv import load_dotenv
from typing import AsyncIterator, Callable, Awaitable

from src.server.utils import amerge

load_dotenv()

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"

# Model events worth forwarding to the browser
EVENTS_TO_BROWSER = {
    "response.audio.delta",
    "response.audio.done",
    "response.audio_transcript.delta",
    "response.audio_transcript.done",
    "conversation.item.input_audio_transcription.completed",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "response.done",
    "error",
}


class OpenAIVoiceReactAgent:
    def __init__(self, model: str, tools: list, instructions: str, api_key: str = None) -> None:
        self.model = model
        self.tools = tools
        self.instructions = instructions
        self.api_key = api_key or os.getenv("SECRET_OPENAI")

    async def aconnect(
        self,
        input_stream: AsyncIterator[str],
        send_output_chunk: Callable[[str], Awaitable[None]],
    ) -> None:
        tools_by_name = {t["name"]: t["func"] for t in self.tools}
        tool_defs = [
            {
                "type": "function",
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            }
            for t in self.tools
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                f"{OPENAI_REALTIME_URL}?model={self.model}", headers=headers
            ) as model_ws:

                async def model_stream() -> AsyncIterator[str]:
                    async for ws_msg in model_ws:
                        if ws_msg.type == aiohttp.WSMsgType.TEXT:
                            yield ws_msg.data
                        else:
                            break

                await model_ws.send_str(json.dumps({
                    "type": "session.update",
                    "session": {
                        "instructions": self.instructions,
                        "voice": "alloy",
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "input_audio_transcription": {"model": "whisper-1"},
                        "turn_detection": {"type": "server_vad"},
                        "tools": tool_defs,
                        "tool_choice": "auto",
                    },
                }))

                try:
                    async for source, raw in amerge(browser=input_stream, model=model_stream()):
                        if source == "browser":
                            # Forward browser events (audio chunks etc.) straight to the model
                            await model_ws.send_str(raw)
                            continue

                        event = json.loads(raw)
                        event_type = event.get("type", "")

                        if event_type == "response.function_call_arguments.done":
                            # Execute the requested RAG tool and hand the result back
                            tool_name = event.get("name")
                            call_id = event.get("call_id")
                            print(f"voice agent tool call -> {tool_name}")
                            func = tools_by_name.get(tool_name)
                            if func is None:
                                result = f"Unknown tool: {tool_name}"
                            else:
                                try:
                                    args = json.loads(event.get("arguments") or "{}")
                                    query = args.get("query", "")
                                    # Pass mode through when the tool accepts it (knowledge_base_search)
                                    if "mode" in args:
                                        result = await asyncio.to_thread(func, query, args["mode"])
                                    else:
                                        result = await asyncio.to_thread(func, query)
                                except Exception as e:
                                    print("ERROR while executing voice tool:", e)
                                    result = "Tool execution failed."

                            await model_ws.send_str(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": str(result),
                                },
                            }))
                            await model_ws.send_str(json.dumps({"type": "response.create"}))

                        elif event_type in EVENTS_TO_BROWSER:
                            await send_output_chunk(raw)
                except Exception as e:
                    print("Voice agent connection closed:", e)
