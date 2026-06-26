from sound.config import SAMPLE_RATE
from sound.preprocess.filters import high_pass_pcm, preemphasis
from sound.preprocess.normalize import rms_normalize_pcm
from sound.preprocess.trim import trim_silence


def preprocess_pcm(pcm):
    pcm = high_pass_pcm(pcm)
    pcm = preemphasis(pcm)
    pcm = trim_silence(pcm)
    if len(pcm) < int(SAMPLE_RATE * 0.08):
        return pcm
    pcm = rms_normalize_pcm(pcm)
    return pcm
