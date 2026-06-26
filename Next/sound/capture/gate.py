import audioop

from sound.config import SAMPLE_RATE


def voice_is_loud_enough(audio, energy_threshold, noise_floor_rms):
    """RMS gate truoc khi vao preprocess/MFCC. Tra (ok, rms, noise_floor_moi)."""
    try:
        raw = audio.get_raw_data(convert_rate=SAMPLE_RATE, convert_width=2)
        rms = audioop.rms(raw, 2)
        loud = rms >= max(energy_threshold, int(noise_floor_rms * 2.2))
        if rms > 0 and not loud:
            noise_floor_rms = int(noise_floor_rms * 0.92 + rms * 0.08)
        return loud, rms, noise_floor_rms
    except Exception:
        return False, 0, noise_floor_rms
