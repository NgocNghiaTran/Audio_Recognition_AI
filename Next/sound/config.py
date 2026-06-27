import os

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_PKG_DIR)

SAMPLE_RATE = 16000
N_MFCC = 13

# Level 1: 5 nguyen am — thu tu guided test U -> I -> O -> E -> A
LABELS = ('U', 'I', 'O', 'E', 'A')
LABEL_TO_COMMAND = {
    'U': 'JUMP',
    'I': 'RIGHT',
    'O': 'LEFT',
    'E': 'RIGHT',   # giong I
    'A': 'LEFT',    # giong O
}
LEGACY_LABEL_ALIASES = {}

TEMPLATES_DIR = os.path.join(_PKG_DIR, 'data', 'templates')
TEMPLATES_PATH = os.path.join(TEMPLATES_DIR, 'templates3.npz')

DEFAULT_THRESHOLD = 0.68
DEFAULT_MARGIN = 0.04
DEFAULT_THRESHOLDS = {label: 0.68 for label in LABELS}

# Adaptive margin: margin tu dong theo do phan tan diem.
# gap nho (nhieu class gan nhau) -> margin nho hon de tranh reject.
# cong thuc: base_margin + temperature * std(scores)
ADAPTIVE_MARGIN_BASE = 0.01   # nghieng tren cung
ADAPTIVE_MARGIN_SCALE = 0.03  # he so nhan voi std

# Per-label-pair margin overrides.
# Neu cap nay lai nhau trong khong gian MFCC, dung margin nho hon.
# Format: (label1, label2) -> margin (unordered)
PAIR_MARGINS = {
    ('A', 'E'): 0.025,   # cosine 0.975 - rat de nham
    ('A', 'I'): 0.025,   # cosine 0.955
    ('A', 'U'): 0.025,   # cosine 0.959
    ('U', 'O'): 0.030,   # cosine 0.931
    ('U', 'E'): 0.025,   # cosine 0.925
}

# Softmax temperature cho p(best > second).
# Temperature cao -> probability giua cac class deu hon.
# Khi co 5 class sat nhau, temperature > 1 giup phan biet hon.
SOFTMAX_TEMPERATURE = 1.2

# Fallback: neu best khong pass margin, thu second-best voi nguong thap hon.
FALLBACK_THRESHOLD_RELAX = 0.04   # nguong second chiu thap hon best
FALLBACK_MARGIN_RELAX = 0.02      # margin second cung the
TRIM_SILENCE_RATIO = 0.12
TARGET_RMS = 0.12
MIN_MFCC_SAMPLES = 4096

CHUNKS_PER_PHASE = 60
# (label, command, chunk_start, chunk_end) — cap nhat khi doi CHUNKS_PER_PHASE
def guided_test_phases(chunks_per_phase=CHUNKS_PER_PHASE):
    order = (
        ('U', 'JUMP'),
        ('I', 'RIGHT'),
        ('O', 'LEFT'),
        ('E', 'RIGHT'),
        ('A', 'LEFT'),
    )
    phases = []
    start = 1
    for label, command in order:
        end = start + chunks_per_phase - 1
        phases.append((label, command, start, end))
        start = end + 1
    return phases


DEFAULT_XLSX = os.path.join(PROJECT_ROOT, 'Data_Speech_Audio.xlsx')

LABEL_MEANINGS = (
    'U=nhay | I=phai | O=trai | E=phai (nhu I) | A=trai (nhu O)'
)
