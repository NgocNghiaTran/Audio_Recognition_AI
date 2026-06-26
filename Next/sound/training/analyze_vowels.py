"""
Phan tich pho / MFCC cac nguyen am — chung minh am nao gan nhau.

Chay:
    cd D:\\AI\\GameClone\\MarioPygame\\Next
    py analyze_vowels.py

Xuat:
    sound/data/analysis/mfcc_similarity.png   — ma tran cosine 5x5
    sound/data/analysis/spectrograms.png      — spectrogram tung am
    sound/data/analysis/mean_spectrum.png     — pho trung binh
    sound/data/analysis/report.txt            — bang so + cap de nham
"""
import os
import sys

# Fix: Dua project folder len truoc de khong import nham package 'sound' trong venv
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from sound.config import LABELS, SAMPLE_RATE, TEMPLATES_DIR
from sound.capture.pcm import read_wav
from sound.matching.cosine import cosine_similarity
from sound.matching.matcher import VoiceMfccMatcher
from sound.preprocess.pipeline import preprocess_pcm

OUT_DIR = os.path.join(TEMPLATES_DIR, '..', 'analysis')


def _load_librosa():
    try:
        import librosa
        return librosa
    except ImportError:
        print('Can: pip install librosa matplotlib')
        sys.exit(1)


def _load_matplotlib():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print('Can: pip install matplotlib')
        sys.exit(1)


def collect_wav_paths(label):
    label_dir = os.path.join(TEMPLATES_DIR, label)
    if not os.path.isdir(label_dir):
        return []
    return sorted(
        os.path.join(label_dir, n) for n in os.listdir(label_dir)
        if n.lower().endswith('.wav'))


def mean_mfcc_vectors(matcher):
    """Vector MFCC trung binh moi am (cung pipeline game)."""
    vectors = {}
    counts = {}
    for label in LABELS:
        vecs = []
        for path in collect_wav_paths(label):
            pcm = read_wav(path)
            vec = matcher._feature_vector(pcm)
            if vec is not None:
                vecs.append(vec)
        if vecs:
            vectors[label] = np.mean(np.stack(vecs, axis=0), axis=0)
            counts[label] = len(vecs)
    return vectors, counts


def similarity_matrix(vectors):
    n = len(LABELS)
    mat = np.zeros((n, n), dtype=np.float32)
    for i, li in enumerate(LABELS):
        for j, lj in enumerate(LABELS):
            if li in vectors and lj in vectors:
                mat[i, j] = cosine_similarity(vectors[li], vectors[lj])
            else:
                mat[i, j] = float('nan')
    return mat


def plot_similarity(mat, path):
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, vmin=0.5, vmax=1.0, cmap='RdYlGn')
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel('MFCC template (cot)')
    ax.set_ylabel('MFCC template (dong)')
    ax.set_title('Cosine similarity giua template 5 nguyen am\n(cao = giong nhau trong khong gian MFCC)')
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, '%.2f' % val, ha='center', va='center', fontsize=10)
    fig.colorbar(im, ax=ax, label='cosine')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pick_example_wav(label):
    paths = collect_wav_paths(label)
    return paths[0] if paths else None


def plot_spectrograms(path):
    librosa = _load_librosa()
    plt = _load_matplotlib()
    fig, axes = plt.subplots(len(LABELS), 1, figsize=(10, 2.2 * len(LABELS)))
    if len(LABELS) == 1:
        axes = [axes]
    for ax, label in zip(axes, LABELS):
        wav = pick_example_wav(label)
        if not wav:
            ax.set_title('%s — khong co mau' % label)
            continue
        pcm = preprocess_pcm(read_wav(wav))
        S = librosa.feature.melspectrogram(y=pcm, sr=SAMPLE_RATE, n_mels=64)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(
            S_db, sr=SAMPLE_RATE, x_axis='time', y_axis='mel', ax=ax)
        ax.set_title('Mel-spectrogram: %s (1 mau dai dien)' % label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mean_spectrum(path):
    librosa = _load_librosa()
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))
    for label in LABELS:
        specs = []
        for wav in collect_wav_paths(label)[:10]:
            pcm = preprocess_pcm(read_wav(wav))
            if len(pcm) < 256:
                continue
            fft = np.abs(np.fft.rfft(pcm))
            freqs = np.fft.rfftfreq(len(pcm), 1.0 / SAMPLE_RATE)
            specs.append(np.interp(np.linspace(0, 4000, 200), freqs, fft))
        if specs:
            mean_spec = np.mean(np.stack(specs), axis=0)
            ax.plot(np.linspace(0, 4000, 200), mean_spec, label=label, linewidth=2)
    ax.set_xlabel('Tan so (Hz)')
    ax.set_ylabel('Bien do (FFT trung binh)')
    ax.set_title('Pho tan so trung binh 5 nguyen am (0-4kHz)\nDuong sat nhau = nang luong pho giong nhau')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 4000)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(mat, counts, path):
    lines = []
    lines.append('=== MFCC cosine similarity (template trung binh) ===')
    lines.append('Cang gan 1.0 = cang de nham trong he thong game.')
    lines.append('')
    header = '     ' + '  '.join('%5s' % x for x in LABELS)
    lines.append(header)
    for i, li in enumerate(LABELS):
        row = '%4s ' % li
        for j in range(len(LABELS)):
            row += '%5.2f ' % mat[i, j]
        lines.append(row)
    lines.append('')
    lines.append('Mau / am: ' + ', '.join('%s=%d' % (k, counts.get(k, 0)) for k in LABELS))
    lines.append('')
    pairs = []
    for i, li in enumerate(LABELS):
        for j, lj in enumerate(LABELS):
            if j <= i:
                continue
            pairs.append((mat[i, j], li, lj))
    pairs.sort(reverse=True)
    lines.append('=== Cap am GAN NHAU nhat (de nham) ===')
    for score, a, b in pairs[:8]:
        lines.append('  %s — %s : cosine = %.3f' % (a, b, score))
    lines.append('')
    lines.append('Giai thich nhanh:')
    lines.append('- Test 7: noi "you" luc phase U -> pho gan E/O hon U thuan.')
    lines.append('- Noi "ay" (ten chu A) luc phase A -> pho gan E hon A thuan.')
    lines.append('- I, O, E doc vowel thuan thi cosine template tach ro hon.')
    text = '\n'.join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    matcher = VoiceMfccMatcher()
    vectors, counts = mean_mfcc_vectors(matcher)
    missing = [l for l in LABELS if l not in vectors]
    if missing:
        print('Thieu mau:', missing)
        sys.exit(1)

    mat = similarity_matrix(vectors)
    sim_path = os.path.join(OUT_DIR, 'mfcc_similarity.png')
    spec_path = os.path.join(OUT_DIR, 'spectrograms.png')
    fft_path = os.path.join(OUT_DIR, 'mean_spectrum.png')
    report_path = os.path.join(OUT_DIR, 'report.txt')

    plot_similarity(mat, sim_path)
    try:
        plot_spectrograms(spec_path)
    except Exception as exc:
        print('Spectrogram skip:', exc)
    plot_mean_spectrum(fft_path)
    write_report(mat, counts, report_path)

    print('')
    print('Da luu:')
    print(' ', sim_path)
    print(' ', spec_path)
    print(' ', fft_path)
    print(' ', report_path)


if __name__ == '__main__':
    main()
