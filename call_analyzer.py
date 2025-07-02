import json
import logging
from typing import List, Optional

from openai.types.chat import ChatCompletionMessageParam
from pipecat.frames.frames import Frame, EndFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.openai import OpenAILLMService, OpenAILLMContext

logger = logging.getLogger(__name__)


class CallAnalyzer(FrameProcessor):
    """
    A processor that collects transcripts during a call and then uses an LLM to:
    1. Extract important data points
    2. Evaluate if call goals were met
    3. Generate a summary of the conversation

    All in a single LLM call at the end of the conversation.
    """

    def __init__(
        self,
        llm_service: OpenAILLMService,
        output_file: Optional[str] = None,
        evaluation_criteria: Optional[List[str]] = None,
    ):
        """
        Initialize the CallAnalyzer.

        Args:
            llm_service: The OpenAI LLM service to use for analysis
            output_file: Optional file path to save the analysis results
            evaluation_criteria: Optional list of criteria to evaluate the call against
        """
        self.llm_service = llm_service
        self.output_file = output_file
        self.evaluation_criteria = evaluation_criteria or [
            "User's name was obtained",
            "User's nationality was obtained",
            "Purpose of the call was explained",
        ]
        self.transcript = []
        self.analysis_complete = False

        super().__init__(name="CallAnalyzer")

    async def process_frame(self, frame: Frame, direction: str):
        """Process incoming frames, collecting transcript and analyzing at the end."""

        print("Processing frame:", frame, direction)

        # Collect transcript entries
        if isinstance(frame, TranscriptionFrame):
            speaker = "User" if frame.user_id != "assistant" else "Assistant"
            self.transcript.append(f"{speaker}: {frame.text}")

        # When call ends, analyze the transcript
        if isinstance(frame, EndFrame) and not self.analysis_complete:
            await self._analyze_transcript()
            self.analysis_complete = True

    async def _analyze_transcript(self) -> None:
        """Send the transcript to the LLM for analysis."""
        if not self.transcript:
            logger.warning("No transcript collected, skipping analysis")
            return

        full_transcript = "\n".join(self.transcript)

        # Create the analysis prompt
        criteria_text = "\n".join(
            [f"- {criterion}" for criterion in self.evaluation_criteria]
        )

        system_prompt = f"""
        You are an expert call analyzer. Given a transcript of a conversation, please:
        
        1. EXTRACT DATA: Extract key information about the user (name, nationality, concerns, preferences, etc.)
        
        2. EVALUATE: Evaluate whether the following criteria were met:
        {criteria_text}
        
        3. SUMMARIZE: Provide a concise summary of the conversation (key points, decisions, action items)
        
        Format your response as a JSON object with the following structure:
        {{
            "extracted_data": {{
                "user_name": "string or null",
                "nationality": "string or null",
                "concerns": ["list of strings"],
                "preferences": ["list of strings"],
                "other_notable_info": ["list of strings"]
            }},
            "evaluation": {{
                "criteria_met": [
                    {{"criterion": "criterion text", "met": true/false, "evidence": "supporting evidence"}}
                ],
                "overall_success": true/false,
                "improvement_suggestions": ["list of suggestions"]
            }},
            "summary": "concise summary of the conversation"
        }}
        """

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Here is the transcript to analyze:\n\n{full_transcript}",
            },
        ]

        try:
            # Create a context object for the LLM call
            context = OpenAILLMContext(messages=messages)

            # Get streaming response from LLM
            response_stream = await self.llm_service.get_chat_completions(
                context, messages
            )

            # Collect the full response from the stream
            response_text = ""
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content

            # Parse the JSON response
            try:
                analysis_results = json.loads(response_text)
                logger.info("Call analysis complete")

                # Save to file if specified
                if self.output_file:
                    with open(self.output_file, "w") as f:
                        json.dump(analysis_results, f, indent=2)
                    logger.info(f"Analysis saved to {self.output_file}")

                return analysis_results

            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON")
                logger.debug(f"Raw response: {response_text}")

        except Exception as e:
            logger.error(f"Error during call analysis: {str(e)}")
