import wave

import numpy as np

from sound.config import SAMPLE_RATE


def pcm_from_audio_data(raw_bytes, sample_rate=SAMPLE_RATE):
    return np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def read_wav(path):
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    pcm = pcm_from_audio_data(raw, sr)
    if sr != SAMPLE_RATE and len(pcm) > 1:
        idx = np.linspace(0, len(pcm) - 1, int(len(pcm) * SAMPLE_RATE / sr))
        pcm = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.float32)
    return pcm
