# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv()

import json
import logging
import os

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import PlainTextResponse
from twilio.rest import Client
from twilio.twiml.voice_response import Connect, VoiceResponse

from bot import run_outbound_bot

# Environment variables as constants
TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID") 
TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN") 
TWILIO_PHONE_NUMBER: str = os.getenv("TWILIO_PHONE_NUMBER") 
SERVER_DOMAIN: str = os.getenv("SERVER_DOMAIN")

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()
logger = logging.getLogger(__name__)

# Twilio client
twilio_client = Client(
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
)



@app.post("/start-outbound-call")
async def start_outbound_call(phone_number: str):
    """Initiate an outbound call to the specified phone number."""
    try:
        twiml = VoiceResponse()
        connect: Connect = twiml.connect() 
        logger.info(f"wss://{SERVER_DOMAIN}/ws")
        connect.stream(
            url=f"wss://{SERVER_DOMAIN}/ws",
            # Pass the target phone number as a parameter
            **{"name": "outbound_stream"},
        )

        # Make the outbound call
        call = twilio_client.calls.create(
            to=phone_number,
            from_=TWILIO_PHONE_NUMBER,
            twiml=str(twiml),
            status_callback=f"https://{SERVER_DOMAIN}/call-status",
            status_callback_event=["initiated", "ringing", "answered", "completed"],
        )

        logger.info(f"Outbound call initiated: {call.sid} to {phone_number}")
        return {"call_sid": call.sid, "status": "initiated"}

    except Exception as e:
        logger.error(f"Failed to start outbound call: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/call-status")
async def call_status_webhook(request: Request):
    """Handle Twilio call status updates."""
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    call_status = form_data.get("CallStatus")

    logger.info(f"Call {call_sid} status: {call_status}")

    # Handle different call statuses
    if call_status == "completed":
        # Clean up any resources
        pass
    elif call_status == "failed":
        # Handle failed calls
        pass

    return PlainTextResponse("OK")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connection from Twilio Media Streams."""
    await websocket.accept()
    logger.info("WebSocket connection established")

    # Read the initial messages from Twilio
    messages = []
    async for message in websocket.iter_text():
        data = json.loads(message)
        messages.append(data)

        # Look for the 'start' event
        if data.get("event") == "start":
            stream_sid = data["start"]["streamSid"]
            call_sid = data["start"]["callSid"]

            logger.info(f"Stream started: {stream_sid}, Call: {call_sid}")

            # Start the bot process
            await run_outbound_bot(websocket, stream_sid, call_sid)
            break


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True)
