import numpy as np

from sound.config import MIN_MFCC_SAMPLES, N_MFCC, SAMPLE_RATE


def load_mfcc_extractor():
    try:
        import librosa

        def extract(y, sr):
            y = np.asarray(y, dtype=np.float32)
            if len(y) < MIN_MFCC_SAMPLES:
                y = np.pad(y, (0, MIN_MFCC_SAMPLES - len(y)), mode='reflect')
            n_fft = min(2048, 1 << int(np.ceil(np.log2(max(len(y), 512)))))
            hop = max(n_fft // 4, 1)
            m = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=hop)
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


def extract_mfcc_mean(pcm, extract_fn, sample_rate=SAMPLE_RATE):
    return extract_fn(pcm, sample_rate)
