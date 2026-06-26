import numpy as np

from sound.config import SAMPLE_RATE


def high_pass_pcm(pcm, cutoff_hz=80.0):
    if len(pcm) < 2:
        return pcm
    alpha = np.exp(-2.0 * np.pi * cutoff_hz / SAMPLE_RATE)
    out = np.empty_like(pcm)
    out[0] = pcm[0]
    for i in range(1, len(pcm)):
        out[i] = alpha * (out[i - 1] + pcm[i] - pcm[i - 1])
    return out


def preemphasis(pcm, coef=0.97):
    if len(pcm) < 2:
        return pcm
    out = np.empty_like(pcm)
    out[0] = pcm[0]
    out[1:] = pcm[1:] - coef * pcm[:-1]
    return out
