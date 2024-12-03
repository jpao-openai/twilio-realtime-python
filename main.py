import os
import json
import base64
import asyncio
import websockets
import aiohttp
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect
from dotenv import load_dotenv
from enum import Enum
from typing import Optional, Callable, List, Dict, Any

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 5050))
SYSTEM_MESSAGE = (
    "You are a helpful and bubbly AI assistant who loves to chat about "
    "anything the user is interested in and is prepared to offer them facts. "
    "Always stay positive.."
)
VOICE = 'alloy'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created'
]

# In-memory storage for set_memory tool
memory_storage = {}

class TurnDetectionMode(Enum):
    SERVER_VAD = "server_vad"
    MANUAL = "manual"

class RealtimeClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-realtime-preview-2024-10-01",
        voice: str = "alloy",
        instructions: str = "You are a helpful assistant",
        temperature: float = 0.8,
        turn_detection_mode: TurnDetectionMode = TurnDetectionMode.SERVER_VAD,
        on_text_delta: Optional[Callable[[str], None]] = None,
        on_audio_delta: Optional[Callable[[bytes], None]] = None,
        on_interrupt: Optional[Callable[[], None]] = None,
        on_input_transcript: Optional[Callable[[str], None]] = None,
        on_output_transcript: Optional[Callable[[str], None]] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.ws = None
        self.on_text_delta = on_text_delta
        self.on_audio_delta = on_audio_delta
        self.on_interrupt = on_interrupt
        self.on_input_transcript = on_input_transcript
        self.on_output_transcript = on_output_transcript
        self.instructions = instructions
        self.temperature = temperature
        self.base_url = "wss://api.openai.com/v1/realtime"
        self.turn_detection_mode = turn_detection_mode
        self._current_response_id = None
        self._current_item_id = None
        self._is_responding = False

    async def connect(self) -> None:
        """Establish WebSocket connection with the Realtime API."""
        url = f"{self.base_url}?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        self.ws = await websockets.connect(url, extra_headers=headers)
        await self.initialize_session()

    async def initialize_session(self) -> None:
        """Control initial session with OpenAI and register tools."""
        tools = [
            {
                "type": "function",
                "name": "handoff_to_agent",
                "description": "Request human intervention.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason_for_handoff": {
                            "type": "string",
                            "description": "Reason for requesting an agent."
                        }
                    },
                    "required": ["reason_for_handoff"]
                }
            },
            {
                "type": "function",
                "name": "get_weather",
                "description": "Retrieves weather info for a given lat/lng.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number", "description": "Latitude."},
                        "lng": {"type": "number", "description": "Longitude."},
                        "location": {"type": "string", "description": "Location name."}
                    },
                    "required": ["lat", "lng", "location"]
                }
            }
        ]

        session_update = {
            "type": "session.update",
            "session": {
                "turn_detection": {"type": self.turn_detection_mode.value},
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "voice": self.voice,
                "instructions": self.instructions,
                "modalities": ["text", "audio"],
                "temperature": self.temperature,
                "tools": tools,
                "tool_choice": "auto",
            }
        }
        print('Sending session update:', json.dumps(session_update))
        await self.ws.send(json.dumps(session_update))

    async def get_weather(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch weather information."""
        lat = parameters.get("lat")
        lng = parameters.get("lng")
        location = parameters.get("location", "the specified location")

        if lat is None or lng is None:
            return {"error": "Latitude and longitude are required."}

        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current_weather=true"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        weather = data.get("current_weather", {})
                        return {
                            "temperature": weather.get("temperature"),
                            "wind_speed": weather.get("windspeed"),
                            "units": {
                                "temperature": "°C",
                                "wind_speed": "m/s"
                            }
                        }
                    else:
                        return {"error": f"Failed to fetch weather (HTTP {response.status})."}
        except Exception as e:
            return {"error": str(e)}

# i dont think this works 'type': 'function_call' 'name': 'get_weather'
# add cursor documentation -> realtime docs
    async def handle_function_call(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if function_name == "get_weather":
            return await self.get_weather(parameters)
        else:
            print(f"Unknown function: {function_name}")
            return {}

    async def handle_messages(self) -> None:
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type")
                if event_type in LOG_EVENT_TYPES:
                    print(f"Received event: {event_type}", event)

                if event_type == "response.done":
                    # Parse the output to find the function call
                    output = event.get("response", {}).get("output", [])
                    for item in output:
                        if item.get("type") == "function_call":
                            function_name = item.get("name")
                            call_arguments = json.loads(item.get("arguments", "{}"))
                            result = await self.handle_function_call(function_name, call_arguments)

                            # Send the function response back to the WebSocket
                            response = {
                                "type": "function.response",
                                "name": function_name,
                                "result": result
                            }
                            await self.ws.send(json.dumps(response))
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")
        except Exception as e:
            print(f"Error: {str(e)}")

app = FastAPI()

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    await websocket.accept()
    client = RealtimeClient(api_key=OPENAI_API_KEY)
    await client.connect()
    await asyncio.gather(client.handle_messages())