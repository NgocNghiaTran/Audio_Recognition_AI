import numpy as np

from sound.config import TARGET_RMS


def rms_normalize_pcm(pcm, target_rms=TARGET_RMS):
    rms = float(np.sqrt(np.mean(pcm * pcm))) if len(pcm) else 0.0
    if rms < 1e-6:
        return pcm
    scaled = pcm * (target_rms / rms)
    peak = float(np.max(np.abs(scaled)))
    if peak > 1.0:
        scaled = scaled / peak
    return scaled
