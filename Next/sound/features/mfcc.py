import numpy as np

from sound.config import MIN_MFCC_SAMPLES, N_MFCC, SAMPLE_RATE


def load_mfcc_extractor(use_delta=False):
    """Load MFCC extractor với tùy chọn Delta MFCC.

    Args:
        use_delta: Nếu True, trả về 39 features (13 MFCC + 13 Delta + 13 Delta-Delta)
                   Nếu False, trả về 13 features (chỉ MFCC)
    """
    try:
        import librosa

        def extract(y, sr):
            y = np.asarray(y, dtype=np.float32)
            if len(y) < MIN_MFCC_SAMPLES:
                y = np.pad(y, (0, MIN_MFCC_SAMPLES - len(y)), mode='reflect')
            n_fft = min(2048, 1 << int(np.ceil(np.log2(max(len(y), 512)))))
            hop = max(n_fft // 4, 1)

            # MFCC cơ bản
            m = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=hop)

            if use_delta:
                # Delta MFCC (thay đổi tốc độ)
                m_delta = librosa.feature.delta(m)
                # Delta-Delta MFCC (gia tốc)
                m_delta2 = librosa.feature.delta(m, order=2)
                # Concatenate: 13 + 13 + 13 = 39 features
                return np.concatenate([
                    m.mean(axis=1),       # MFCC tĩnh
                    m_delta.mean(axis=1),  # Delta (động)
                    m_delta2.mean(axis=1) # Delta-Delta (gia tốc)
                ])
            else:
                return m.mean(axis=1)

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


def extract_mfcc_mean(pcm, extract_fn, sample_rate=SAMPLE_RATE, use_delta=False):
    return extract_fn(pcm, sample_rate)
