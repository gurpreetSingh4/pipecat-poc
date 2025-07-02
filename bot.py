import asyncio
import datetime as dt
import logging
import os
import io
import wave
from fastapi import WebSocket
from openai.types.chat import ChatCompletionMessageParam
from pipecat.frames.frames import EndFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMContext, OpenAILLMService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from call_analyzer import CallAnalyzer

# Environment variables as constants
TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID")  # type: ignore
TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN")  # type: ignore
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")  # type: ignore
DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY")  # type: ignore
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY")  # type: ignore
ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID")  # type: ignore


async def run_outbound_bot(websocket: WebSocket, stream_sid: str, call_sid: str):
    """Run the Pipecat bot for outbound call with recording and analysis."""
    logger = logging.getLogger(__name__)

    # Create directories for recordings and analysis
    os.makedirs("recordings", exist_ok=True)
    os.makedirs("transcripts", exist_ok=True)
    os.makedirs("analysis", exist_ok=True)

    # Generate unique filenames based on call_sid and timestamp
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_file = f"transcripts/call_{call_sid}_{timestamp}.txt"
    analysis_file = f"analysis/call_{call_sid}_{timestamp}.json"

    # Initialize Twilio serializer
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=TWILIO_ACCOUNT_SID,
        auth_token=TWILIO_AUTH_TOKEN,
    )

    # Create WebSocket transport
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    # Initialize AI services
    llm = OpenAILLMService(
        api_key=OPENAI_API_KEY,
        model="gpt-4o",
    )

    # Configure STT to pass through audio for recording
    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        model="nova-2",
        language="en",
        sample_rate=8000,
        audio_passthrough=True,  # Enable audio passthrough for recording
    )

    tts = ElevenLabsTTSService(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ELEVENLABS_VOICE_ID,
        sample_rate=8000,
        # params=ElevenLabsTTSService.InputParams(
        #     stability=0.7,
        #     similarity_boost=0.8,
        #     style=0.5,
        #     use_speaker_boost=True
        # )
    )

    # Create conversation context
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": """You are Alexa. You are making an outbound call. Introduce yourself politely and ask for the user's name and nationality. Keep your responses short and conversational.""",
        }
    ]

    # Set up context aggregator using OpenAILLMContext
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # Initialize audio buffer processor for recording
    audio_buffer = AudioBufferProcessor(
        sample_rate=8000,  # Match Twilio's 8kHz sample rate
        num_channels=1,
        buffer_size=0,  # Only trigger when recording stops
    )

    # Initialize transcript processor
    transcript = TranscriptProcessor()

    # Initialize call analyzer with evaluation criteria
    evaluation_criteria = [
        "User's name was obtained",
        "User's nationality was obtained",
    ]

    call_analyzer = CallAnalyzer(
        llm_service=llm,
        output_file=analysis_file,
        evaluation_criteria=evaluation_criteria,
    )

    # Build the pipeline with recording and analysis components
    pipeline = Pipeline(
        [
            transport.input(),  # WebSocket input from Twilio
            stt,  # Speech-to-text (with audio_passthrough=True)
            transcript.user(),  # Capture user transcripts
            context_aggregator.user(),  # Add user message to context
            llm,  # Language model processing
            tts,  # Text-to-speech
            transport.output(),  # WebSocket output to Twilio
            transcript.assistant(),  # Capture assistant transcripts
            context_aggregator.assistant(),  # Add assistant response to context
            audio_buffer,  # Record audio from the conversation
            call_analyzer,  # Analyze the call at the end
        ]
    )

    # Handle transcript updates and save to file
    @transcript.event_handler("on_transcript_update")
    async def handle_transcript_update(processor, frame):
        # Create transcripts directory if it doesn't exist
        os.makedirs("transcripts", exist_ok=True)

        # Append transcript to file
        with open(transcript_file, "a") as file:
            for message in frame.messages:
                file.write(f"[{message.timestamp}] {message.role}: {message.content}\n")

        logger.info(f"Transcript updated in {transcript_file}")

    # Define audio recording handler
    @audio_buffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio: bytes, sample_rate: int, num_channels: int):
        if len(audio) > 0:
            # Create recordings directory if it doesn't exist
            os.makedirs("recordings", exist_ok=True)

            # Generate filename with timestamp and call_sid
            filename = f"recordings/call_{call_sid}_{timestamp}.wav"

            # Save audio to WAV file
            with io.BytesIO() as buffer:
                with wave.open(buffer, "wb") as wf:
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setnchannels(num_channels)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio)

                # Write to file
                with open(filename, "wb") as file:
                    file.write(buffer.getvalue())

            logger.info(f"Conversation audio saved to {filename}")

    # Start recording when the call connects
    @transport.event_handler("on_client_connected")
    async def on_client_connected_start_recording(transport, client):
        await audio_buffer.start_recording()
        logger.info("Audio recording started")

    # Stop recording when the call ends
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected_stop_recording(transport, client):
        await audio_buffer.stop_recording()
        logger.info("Audio recording stopped")

    # PipelineParams
    # Create pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,  # Twilio uses 8kHz
            audio_out_sample_rate=8000,
            allow_interruptions=True,
        ),
    )

    # Event handlers
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        """Called when the call is connected."""
        logger.info("Call connected, starting conversation")
        await asyncio.sleep(1)  # Wait a moment for the connection to stabilize

        logger.info("Queueing initial message to LLM via TranscriptionFrame")
        transcription = TranscriptionFrame(
            text="The call has been answered. Please introduce yourself and state the purpose of your call.",
            user_id="system_initiator",
            timestamp=dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        )
        # context_aggregator.user() is at index 2 in the pipeline:
        # [transport.input(), stt, context_aggregator.user(), llm, ...]
        try:
            await task.queue_frames([transcription])
            logger.info("Initial message pushed to context_aggregator.user()")
        except Exception as e:
            logger.error(
                f"Error pushing initial message to context_aggregator.user(): {e}"
            )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        """Called when the call ends."""
        logger.info("Call disconnected")
        await task.queue_frames([EndFrame()])

    # Run the pipeline
    logger = logging.getLogger(__name__)
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
