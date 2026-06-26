import os

import numpy as np

from sound.config import (
    PAIR_MARGINS,
    ADAPTIVE_MARGIN_BASE,
    ADAPTIVE_MARGIN_SCALE,
    SOFTMAX_TEMPERATURE,
    FALLBACK_THRESHOLD_RELAX,
    FALLBACK_MARGIN_RELAX,
    DEFAULT_MARGIN,
    DEFAULT_THRESHOLD,
    DEFAULT_THRESHOLDS,
)


def env_float(name, default):
    raw = os.environ.get(name, '').strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def passes_threshold(best_score, best_label, thresholds, default_threshold):
    return best_score >= thresholds.get(best_label, default_threshold)


def passes_margin(best_score, second_score, margin=DEFAULT_MARGIN):
    if margin <= 0:
        return True
    return (best_score - second_score) >= margin


def _softmax_scores(scores, temperature=SOFTMAX_TEMPERATURE):
    """Tra ve softmax probability giua cac class. Temperature > 1 lam cac class deu hon."""
    vals = np.array(list(scores.values()), dtype=np.float64)
    vals = vals / temperature
    vals -= np.max(vals)
    exp = np.exp(vals)
    prob = exp / (np.sum(exp) + 1e-12)
    labels = list(scores.keys())
    return {l: float(prob[i]) for i, l in enumerate(labels)}


def softmax_probability(scores):
    return _softmax_scores(scores, temperature=SOFTMAX_TEMPERATURE)


def adaptive_margin(scores_dict):
    """Tinh margin tu dong theo do phan tan diem giua cac class.

    Neu cac class rat gan nhau (std nho) -> margin nho.
    Neu cac class xa nhau (std lon) -> margin lon.

    Cong thuc: base + scale * std(scores)
    """
    scores = np.array(list(scores_dict.values()), dtype=np.float32)
    std = float(np.std(scores))
    margin = ADAPTIVE_MARGIN_BASE + ADAPTIVE_MARGIN_SCALE * std
    return max(0.005, min(margin, 0.10))


def pair_margin(labels, scores):
    """Tra ve margin nho nhat trong cac cap (best, others) neu cap do co override."""
    best_label = max(scores, key=scores.get)
    pair_key = tuple(sorted([best_label, '']))  # placeholder
    min_margin = None
    for other_label, other_score in scores.items():
        if other_label == best_label:
            continue
        pair = tuple(sorted([best_label, other_label]))
        override = PAIR_MARGINS.get(pair)
        m = override if override is not None else DEFAULT_MARGIN
        if min_margin is None or m < min_margin:
            min_margin = m
    return min_margin if min_margin is not None else DEFAULT_MARGIN


def classify_with_adaptive(scores, thresholds=DEFAULT_THRESHOLDS,
                           default_threshold=DEFAULT_THRESHOLD,
                           fallback_threshold_relax=FALLBACK_THRESHOLD_RELAX,
                           fallback_margin_relax=FALLBACK_MARGIN_RELAX,
                           use_margin=True):
    """Phan loaji voi adaptive margin + softmax probability.

    Tra ve (label, best_score, all_scores) hoac (None, best_score, scores)
    neu reject.

    Strategy (use_margin=True):
    1. Tinh adaptive margin tu std(scores)
    2. Neu best - second >= adaptive_margin: ACCEPT
    3. Neu best - second < adaptive_margin nhung prob(best)/prob(second) > 2: ACCEPT
    4. Neu best khong pass threshold: thu second voi threshold relax
    5. Nguoc lai: REJECT

    Strategy (use_margin=False - RECOMMENDED):
    - Chi can best_score >= threshold -> ACCEPT
    - Khong can margin check vi threshold 0.68 da du loai noise
    """
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = ranked[0]
    second_label, second_score = ranked[1]
    scores_with_prob = _softmax_scores(scores)

    # 1. Check threshold
    thresh = thresholds.get(best_label, default_threshold)
    if best_score < thresh:
        # 4. Fallback: thu second-best voi threshold relax
        second_thresh = thresholds.get(second_label, default_threshold) - fallback_threshold_relax
        if (second_score >= second_thresh and
                (second_score - ranked[2][1] if len(ranked) > 2 else 0) >= fallback_margin_relax):
            return second_label, second_score, scores
        scores['_reject'] = 'threshold|best=%s' % best_label
        return None, best_score, scores

    # 2. Margin check - BO qua neu use_margin=False
    if not use_margin:
        return best_label, best_score, scores

    adapt_m = adaptive_margin(scores)
    raw_margin = best_score - second_score

    if raw_margin >= adapt_m:
        return best_label, best_score, scores

    # 3. Softmax probability ratio
    prob_best = scores_with_prob.get(best_label, 0)
    prob_second = scores_with_prob.get(second_label, 0)
    if prob_second > 0 and (prob_best / prob_second) > 2.0:
        return best_label, best_score, scores

    # 5. Reject
    scores['_reject'] = 'margin|best=%s|adapt_m=%.3f|raw=%.3f|prob_ratio=%.2f' % (
        best_label, adapt_m, raw_margin,
        prob_best / prob_second if prob_second > 0 else 999)
    return None, best_score, scores
