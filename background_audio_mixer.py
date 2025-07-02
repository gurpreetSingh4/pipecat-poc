import wave
import numpy as np
from pipecat.frames.frames import AudioFrame
from pipecat.processors.frame_processor import FrameProcessor

class BackgroundAudioMixer(FrameProcessor):
    def __init__(self, background_audio_path: str, volume: float = 0.2):
        super().__init__(name="BackgroundAudioMixer")
        self.bg_audio_data = self._load_audio(background_audio_path)
        self.volume = volume
        self.bg_index = 0

    def _load_audio(self, path):
        with wave.open(path, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16)
        return samples

    async def process_frame(self, frame: AudioFrame, direction: str):
        if not isinstance(frame, AudioFrame) or direction != "output":
            return frame

        audio_samples = np.frombuffer(frame.audio, dtype=np.int16)
        length = len(audio_samples)

        bg_segment = self.bg_audio_data[self.bg_index:self.bg_index + length]
        if len(bg_segment) < length:
            self.bg_index = 0
            bg_segment = self.bg_audio_data[:length]

        self.bg_index += length

        mixed = audio_samples + (bg_segment * self.volume).astype(np.int16)
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)

        frame.audio = mixed.tobytes()
        return frame
