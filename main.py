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
SHOW_TIMING_MATH = False

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
        # Track current response state
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

    async def send_text(self, text: str) -> None:
        """Send text message to the API."""
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": text
                }]
            }
        }
        await self.ws.send(json.dumps(event))
        await self.create_response()

    async def handle_messages(self) -> None:
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type")
                if event_type in LOG_EVENT_TYPES:
                    print(f"Received event: {event_type}", event)
                if event_type == "response.text.delta" and self.on_text_delta:
                    self.on_text_delta(event["delta"])
                elif event_type == "response.audio.delta" and self.on_audio_delta:
                    audio_bytes = base64.b64decode(event["delta"])
                    self.on_audio_delta(audio_bytes)
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")
        except Exception as e:
            print(f"Error in message handling: {str(e)}")

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self.ws:
            await self.ws.close()

app = FastAPI()

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    await websocket.accept()

    client = RealtimeClient(api_key=OPENAI_API_KEY)
    await client.connect()

    async def receive_from_twilio():
        """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
        try:
            async for message in websocket.iter_text():
                data = json.loads(message)
                if data['event'] == 'media' and client.ws.open:
                    latest_media_timestamp = int(data['media']['timestamp'])
                    audio_append = {
                        "type": "input_audio_buffer.append",
                        "audio": data['media']['payload']
                    }
                    await client.ws.send(json.dumps(audio_append))
                elif data['event'] == 'start':
                    print(f"Incoming stream has started {data['start']['streamSid']}")
        except WebSocketDisconnect:
            print("Client disconnected.")
            await client.close()

    async def send_to_twilio():
        await client.handle_messages()

    await asyncio.gather(receive_from_twilio(), send_to_twilio())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
