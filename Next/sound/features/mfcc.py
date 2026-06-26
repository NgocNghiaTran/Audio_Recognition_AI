import numpy as np

from sound.config import MIN_MFCC_SAMPLES, N_MFCC, SAMPLE_RATE

USE_PITCH = True


def load_mfcc_extractor(use_delta=False, use_pitch=None):
    """Load MFCC extractor voi tuy chon Delta MFCC va Pitch.

    Args:
        use_delta: Neu True, them Delta + Delta-Delta MFCC
        use_pitch: Neu True, them pitch features (F0, delta, range)
    """
    if use_pitch is None:
        use_pitch = USE_PITCH

    try:
        import librosa

        def extract(y, sr):
            y = np.asarray(y, dtype=np.float32)
            if len(y) < MIN_MFCC_SAMPLES:
                y = np.pad(y, (0, MIN_MFCC_SAMPLES - len(y)), mode='reflect')
            n_fft = min(2048, 1 << int(np.ceil(np.log2(max(len(y), 512)))))
            hop = max(n_fft // 4, 1)

            features = []

            # 1. MFCC co ban (13 features)
            m = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=hop)
            features.extend(m.mean(axis=1))

            if use_delta:
                # 2. Delta MFCC (13 features)
                m_delta = librosa.feature.delta(m)
                features.extend(m_delta.mean(axis=1))
                # 3. Delta-Delta MFCC (13 features)
                m_delta2 = librosa.feature.delta(m, order=2)
                features.extend(m_delta2.mean(axis=1))

            if use_pitch:
                # 4. Enhanced Pitch (F0) features
                try:
                    f0, voiced_flag, voiced_probs = librosa.pyin(
                        y, fmin=librosa.note_to_hz('C2'),
                        fmax=librosa.note_to_hz('C7'),
                        sr=sr
                    )
                    f0_valid = f0[~np.isnan(f0)]

                    if len(f0_valid) > 5:
                        pitch_mean = np.nanmean(f0)
                        pitch_std = np.nanstd(f0)
                        pitch_min = np.nanmin(f0)
                        pitch_max = np.nanmax(f0)
                        pitch_range = pitch_max - pitch_min
                        voiced_ratio = np.mean(voiced_flag)

                        # Pitch contour: chia thanh 3 phan (start, mid, end)
                        n = len(f0)
                        p_start = np.nanmean(f0[:n//3]) if n >= 3 else pitch_mean
                        p_mid = np.nanmean(f0[n//3:2*n//3]) if n >= 3 else pitch_mean
                        p_end = np.nanmean(f0[2*n//3:]) if n >= 3 else pitch_mean

                        # Pitch slope (do doc: start -> end)
                        if p_end > 0 and p_start > 0:
                            pitch_slope = (p_end - p_start) / p_start
                        else:
                            pitch_slope = 0

                        # Normalized pitch (giam anh huong cua cao do tuyet doi)
                        pitch_norm_mean = pitch_mean / (pitch_mean + 1e-6)
                        pitch_norm_range = pitch_range / (pitch_mean + 1e-6)

                        features.extend([
                            pitch_mean,
                            pitch_std,
                            pitch_range,
                            voiced_ratio,
                            p_start,
                            p_mid,
                            p_end,
                            pitch_slope,
                            pitch_norm_mean,
                            pitch_norm_range,
                        ])
                    else:
                        features.extend([0] * 10)
                except Exception:
                    features.extend([0] * 10)

            return np.array(features, dtype=np.float32)

        return extract, 'librosa'
    except ImportError:
        try:
            from python_speech_features import mfcc as psf_mfcc

            def extract(y, sr):
                m = psf_mfcc(y, samplerate=sr, numcep=N_MFCC)
                return np.mean(m, axis=0)

            return extract, 'python_speech_features'
        except ImportError:
            return None, 'none'


def extract_mfcc_mean(pcm, extract_fn, sample_rate=SAMPLE_RATE, use_delta=False, use_pitch=None):
    return extract_fn(pcm, sample_rate)
