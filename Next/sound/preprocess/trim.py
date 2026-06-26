import numpy as np

from sound.config import SAMPLE_RATE, TRIM_SILENCE_RATIO


def trim_silence(pcm, frame_ms=20, ratio=TRIM_SILENCE_RATIO):
    frame = max(int(SAMPLE_RATE * frame_ms / 1000), 1)
    if len(pcm) < frame:
        return pcm
    energies = []
    for i in range(0, len(pcm) - frame + 1, frame):
        chunk = pcm[i:i + frame]
        energies.append((i, float(np.sqrt(np.mean(chunk * chunk)))))
    if not energies:
        return pcm
    peak = max(e for _, e in energies)
    thresh = max(peak * ratio, 1e-5)
    active = [i for i, e in energies if e >= thresh]
    if not active:
        return pcm
    start = active[0]
    end = min(active[-1] + frame, len(pcm))
    return pcm[start:end]
