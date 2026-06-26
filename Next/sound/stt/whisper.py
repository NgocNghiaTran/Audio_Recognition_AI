import threading

from sound.config import SAMPLE_RATE


class WhisperTranscriber(object):
    def __init__(self, model_name='tiny.en'):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is not None:
                return self._model
            from faster_whisper import WhisperModel
            print('[VOICE] Loading Whisper model "%s" (lần đầu có thể hơi lâu)...' % self.model_name)
            self._model = WhisperModel(self.model_name, device='cpu', compute_type='int8')
            return self._model

    def transcribe(self, audio):
        import numpy as np
        raw = audio.get_raw_data(convert_rate=SAMPLE_RATE, convert_width=2)
        if len(raw) < 3200:
            return ''
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return self._do_transcribe(pcm)

    def transcribe_from_pcm(self, pcm):
        """Transcribe trực tiếp từ PCM array."""
        if len(pcm) < SAMPLE_RATE * 0.05:  # < 50ms
            return ''
        return self._do_transcribe(pcm)

    def _do_transcribe(self, pcm):
        """Transcribe pcm array."""
        segs, _ = self._get_model().transcribe(
            pcm, language='en', beam_size=1, best_of=1, vad_filter=True,
            condition_on_previous_text=False, without_timestamps=True)
        return ' '.join(s.text.strip() for s in segs if s.text.strip())
