import os


def probe_open_microphone(sr, device_index):
    mic = sr.Microphone(device_index=device_index) if device_index is not None else sr.Microphone()
    with mic as source:
        pass
    return mic


def pick_working_microphone(sr):
    env = os.environ.get('MARIO_VOICE_MIC_INDEX', '').strip()
    if env.isdigit():
        try:
            return probe_open_microphone(sr, int(env)), 'index %s (MARIO_VOICE_MIC_INDEX)' % env
        except Exception:
            print('[VOICE] MARIO_VOICE_MIC_INDEX=%s không dùng được, thử mic khác...' % env)
    try:
        names = sr.Microphone.list_microphone_names()
    except Exception:
        names = []
    for idx, lbl in [(None, 'default')] + [(i, '%s (index %d)' % (n, i)) for i, n in enumerate(names) if n]:
        try:
            return probe_open_microphone(sr, idx), lbl
        except Exception:
            continue
    return None, ''
