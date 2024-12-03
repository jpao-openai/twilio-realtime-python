import os
import json
import base64
import asyncio
import websockets
import aiohttp
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from dotenv import load_dotenv

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

    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as openai_ws:
        await initialize_session(openai_ws)

        # Connection specific state
        stream_sid = None
        latest_media_timestamp = 0
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None
        
        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid, latest_media_timestamp
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.open:
                        latest_media_timestamp = int(data['media']['timestamp'])
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        print(f"Incoming stream has started {stream_sid}")
                        response_start_timestamp_twilio = None
                        latest_media_timestamp = 0
                        last_assistant_item = None
                    elif data['event'] == 'mark':
                        if mark_queue:
                            mark_queue.pop(0)
            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            """Receive events from OpenAI Realtime API and handle tool calls."""
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response['type'] in LOG_EVENT_TYPES:
                        print(f"Received event: {response['type']}", response)

                    # Handle tools
                    if response.get('type') == 'tool_call':
                        tool_name = response['tool']['name']
                        parameters = response['tool']['parameters']

                        if tool_name == 'set_memory':
                            key = parameters['key']
                            value = parameters['value']
                            # Simulate memory storage
                            memory_storage[key] = value
                            result = {"ok": True}

                        elif tool_name == 'handoff_to_agent':
                            reason = parameters['reason_for_handoff']
                            result = {"handed_off_to_agent": True}

                        elif tool_name == 'get_weather':
                            lat, lng, location = parameters['lat'], parameters['lng'], parameters['location']
                            result = await fetch_weather(lat, lng, location)

                        else:
                            result = {"error": "Unknown tool"}

                        # Send the tool's result back
                        tool_result_event = {
                            "type": "tool_result",
                            "tool": {"name": tool_name},
                            "result": result
                        }
                        await openai_ws.send(json.dumps(tool_result_event))

                    # Handle other OpenAI events (audio responses, etc.)
                    if response.get('type') == 'response.audio.delta' and 'delta' in response:
                        audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                        audio_delta = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": audio_payload}
                        }
                        await websocket.send_json(audio_delta)

            except Exception as e:
                print(f"Error in send_to_twilio: {e}")


        async def handle_speech_started_event():
            """Handle interruption when the caller's speech starts."""
            nonlocal response_start_timestamp_twilio, last_assistant_item
            print("Handling speech started event.")
            if mark_queue and response_start_timestamp_twilio is not None:
                elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
                if SHOW_TIMING_MATH:
                    print(f"Calculating elapsed time for truncation: {latest_media_timestamp} - {response_start_timestamp_twilio} = {elapsed_time}ms")

                if last_assistant_item:
                    if SHOW_TIMING_MATH:
                        print(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms")

                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed_time
                    }
                    await openai_ws.send(json.dumps(truncate_event))

                await websocket.send_json({
                    "event": "clear",
                    "streamSid": stream_sid
                })

                mark_queue.clear()
                last_assistant_item = None
                response_start_timestamp_twilio = None

        async def send_mark(connection, stream_sid):
            if stream_sid:
                mark_event = {
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": "responsePart"}
                }
                await connection.send_json(mark_event)
                mark_queue.append('responsePart')

        await asyncio.gather(receive_from_twilio(), send_to_twilio())

async def send_initial_conversation_item(openai_ws):
    """Send initial conversation item if AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Greet the user with 'Ahoy! How can I help you?'"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))


async def fetch_weather(lat, lng, location):
    """Fetch weather data from an external API."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current_weather=true"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "location": location,
                    "temperature": data["current_weather"]["temperature"],
                    "wind_speed": data["current_weather"]["windspeed"]
                }
            else:
                return {"error": "Failed to fetch weather data"}

async def initialize_session(openai_ws):
    """Control initial session with OpenAI and register tools."""
    # Define tools
    tools = [
        {
            "type": "tool",
            "name": "set_memory",
            "description": "Saves important data about the user into memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key of the memory value."
                    },
                    "value": {
                        "type": "string",
                        "description": "The value to store."
                    }
                },
                "required": ["key", "value"]
            }
        },
        {
            "type": "tool",
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
            "type": "tool",
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
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
            "tools": tools,
            "tool_choice": "auto",
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    for tool in tools:
        await openai_ws.send(json.dumps(tool))

    # Uncomment the next line to have the AI speak first
    # await send_initial_conversation_item(openai_ws)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
