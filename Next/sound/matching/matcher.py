import os

import numpy as np

from sound.capture.pcm import pcm_from_audio_data, read_wav
from sound.config import (
    ADAPTIVE_MARGIN_BASE,
    ADAPTIVE_MARGIN_SCALE,
    DEFAULT_THRESHOLD,
    DEFAULT_THRESHOLDS,
    LABELS,
    LABEL_TO_COMMAND,
    LEGACY_LABEL_ALIASES,
    SAMPLE_RATE,
    TEMPLATES_DIR,
    TEMPLATES_PATH,
    SOFTMAX_TEMPERATURE,
    USE_DELTA_MFCC,
)
from sound.features.mfcc import load_mfcc_extractor
from sound.matching.cosine import (
    cosine_similarity,
    classify_with_adaptive,
    env_float,
)
from sound.preprocess.pipeline import preprocess_pcm


class VoiceMfccMatcher(object):
    def __init__(self, threshold=None, use_margin=False):
        self.threshold = threshold
        self.use_margin = use_margin
        if self.threshold is None:
            env = os.environ.get('MARIO_MFCC_THRESHOLD', '').strip()
            self.threshold = float(env) if env else DEFAULT_THRESHOLD
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        for label in LABELS:
            key = 'MARIO_MFCC_THRESHOLD_%s' % label
            val = os.environ.get(key, '').strip()
            if val:
                try:
                    self.thresholds[label] = float(val)
                except ValueError:
                    pass
        self._extract, self._backend = load_mfcc_extractor(use_delta=USE_DELTA_MFCC)
        self.templates = {}
        self.enabled = False
        if self._extract is None:
            print('[MFCC] Disabled. pip install librosa  (hoac python_speech_features)')
            return
        if os.path.isfile(TEMPLATES_PATH):
            self._load_templates()
        else:
            self.build_templates()

    def _feature_vector(self, pcm):
        pcm = preprocess_pcm(pcm)
        if len(pcm) < SAMPLE_RATE * 0.05:
            return None
        return self._extract(pcm, SAMPLE_RATE)

    def _label_sample_dir(self, label):
        label_dir = os.path.join(TEMPLATES_DIR, label)
        if os.path.isdir(label_dir):
            return label_dir
        for legacy in LEGACY_LABEL_ALIASES.get(label, ()):
            legacy_dir = os.path.join(TEMPLATES_DIR, legacy)
            if os.path.isdir(legacy_dir):
                print('[MFCC] Dung mau cu "%s/" cho "%s"' % (legacy, label))
                return legacy_dir
        return None

    def build_templates(self):
        if self._extract is None:
            return False
        built = {}
        counts = {}
        for label in LABELS:
            label_dir = self._label_sample_dir(label)
            vectors = []
            if not label_dir:
                continue
            for name in sorted(os.listdir(label_dir)):
                if not name.lower().endswith('.wav'):
                    continue
                vec = self._feature_vector(read_wav(os.path.join(label_dir, name)))
                if vec is not None:
                    vectors.append(vec)
            if vectors:
                built[label] = np.mean(np.stack(vectors, axis=0), axis=0)
                counts[label] = len(vectors)
        if len(built) < len(LABELS):
            print('[MFCC] Thieu mau. Can du U/I/O/E/A trong', TEMPLATES_DIR)
            return False
        os.makedirs(TEMPLATES_DIR, exist_ok=True)
        np.savez(TEMPLATES_PATH, **built)
        self.templates = built
        self.enabled = True
        print('[MFCC] Built templates:', counts, '->', TEMPLATES_PATH)
        print('[MFCC] Backend:', self._backend, '| thresholds:', self.thresholds,
              '| adaptive_margin base=%.3f scale=%.3f' % (ADAPTIVE_MARGIN_BASE, ADAPTIVE_MARGIN_SCALE))
        return True

    def _load_templates(self):
        data = np.load(TEMPLATES_PATH)
        for label in LABELS:
            if label in data.files:
                self.templates[label] = data[label]
            elif ('template_%s' % label) in data.files:
                self.templates[label] = data['template_%s' % label]
            else:
                for legacy in LEGACY_LABEL_ALIASES.get(label, ()):
                    if legacy in data.files:
                        self.templates[label] = data[legacy]
                        print('[MFCC] Dung template cu "%s" cho "%s"' % (legacy, label))
                        break
                    legacy_key = 'template_%s' % legacy
                    if legacy_key in data.files:
                        self.templates[label] = data[legacy_key]
                        print('[MFCC] Dung template cu "%s" cho "%s"' % (legacy_key, label))
                        break
        if 'threshold' in data.files and self.threshold == DEFAULT_THRESHOLD:
            try:
                self.threshold = float(data['threshold'])
            except (TypeError, ValueError):
                pass
        self.enabled = len(self.templates) == len(LABELS)
        if self.enabled:
            print('[MFCC] Loaded templates:', TEMPLATES_PATH)
            print('[MFCC] Backend:', self._backend, '| thresholds:', self.thresholds,
                  '| adaptive_margin base=%.3f scale=%.3f' % (ADAPTIVE_MARGIN_BASE, ADAPTIVE_MARGIN_SCALE))
        else:
            print('[MFCC] Khong du template U/I/O/E/A, build lai ->', TEMPLATES_PATH)
            self.build_templates()

    def classify_pcm(self, pcm):
        if not self.enabled:
            return None, 0.0, {}
        vec = self._feature_vector(pcm)
        if vec is None:
            return None, 0.0, {}
        scores = {label: cosine_similarity(vec, self.templates[label]) for label in LABELS}
        return classify_with_adaptive(scores, self.thresholds, self.threshold,
                                      use_margin=self.use_margin)

    def classify_audio(self, audio):
        raw = audio.get_raw_data(convert_rate=SAMPLE_RATE, convert_width=2)
        pcm = pcm_from_audio_data(raw)
        return self.classify_pcm(pcm)

    def classify_audio_from_pcm(self, pcm):
        return self.classify_pcm(pcm)

    def command_for_label(self, label):
        return LABEL_TO_COMMAND.get(label, '')
