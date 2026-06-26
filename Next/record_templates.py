"""Thu mau am cho template: chi dinh so luong muon thu cho tung am.
Usage:
  python record_templates.py              # Thu tat ca am (50 mau/moi am)
  python record_templates.py --sound U     # Chi thu am U
  python record_templates.py --count 30    # Thu 30 mau cho moi am
  python record_templates.py --all          # Hien thi so luong hien co
"""
import audioop
import argparse
import os
import sys
import wave

from sound.capture.mic import pick_working_microphone
from sound.config import SAMPLE_RATE, TEMPLATES_DIR

RECORD_SECONDS = 1.5
DEFAULT_COUNT = 50

SOUNDS = ['U', 'I', 'O', 'E', 'A']
SOUND_MEANINGS = {
    'U': 'Jump (nhay)',
    'I': 'Right (phai)',
    'O': 'Left (trai)',
    'E': 'Right (phai)',
    'A': 'Left (trai)',
}


def save_wav(audio, path):
    raw = audio.get_raw_data(convert_rate=SAMPLE_RATE, convert_width=2)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw)


def rms_of_audio(audio):
    raw = audio.get_raw_data(convert_rate=SAMPLE_RATE, convert_width=2)
    return audioop.rms(raw, 2)


def next_index(label_dir, label):
    if not os.path.isdir(label_dir):
        return 1
    nums = []
    for name in os.listdir(label_dir):
        if name.startswith(label + '_') and name.endswith('.wav'):
            try:
                nums.append(int(name[len(label) + 1:-4]))
            except ValueError:
                pass
    return max(nums, default=0) + 1


def count_samples(label_dir, label):
    if not os.path.isdir(label_dir):
        return 0
    return sum(
        1 for name in os.listdir(label_dir)
        if name.startswith(label + '_') and name.endswith('.wav'))


def record_one(recognizer, mic, label, index, total):
    label_dir = os.path.join(TEMPLATES_DIR, label)
    path = os.path.join(label_dir, '%s_%03d.wav' % (label, index))
    print('')
    print('--- [%s] Mau %d/%d ---' % (label, index, total))
    print('Nhan ENTER, roi noi am "%s" trong %.1f giay.' % (label, RECORD_SECONDS))
    try:
        input('>> ')
    except EOFError:
        return 'quit'
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.25)
        print('Dang thu... (%.1fs)' % RECORD_SECONDS)
        audio = recognizer.record(source, duration=RECORD_SECONDS)
    rms = rms_of_audio(audio)
    save_wav(audio, path)
    print('Da luu: %s  |  RMS=%d' % (path, rms))
    while True:
        ans = input('Enter=tiep | r=thu lai | s=bo qua | q=thoat: ').strip().lower()
        if ans in ('', 'y'):
            return 'ok'
        if ans == 'r':
            if os.path.isfile(path):
                os.remove(path)
            return record_one(recognizer, mic, label, index, total)
        if ans == 's':
            if os.path.isfile(path):
                os.remove(path)
            return 'skip'
        if ans == 'q':
            if os.path.isfile(path):
                os.remove(path)
            return 'quit'


def show_status(sounds, counts, target):
    print('')
    print('=== TIEN DO ===')
    print('%-10s %-15s %-15s %-10s' % ('Am', 'Hien co', 'Muc tieu', 'Trang thai'))
    print('-' * 50)
    for sound in sounds:
        status = 'OK' if counts[sound] >= target else ''
        print('%-10s %-15d %-15d %-10s' % (sound, counts[sound], target, status))
    print('')


def show_all_status():
    print('')
    print('=== SO LUONG MAU HIEN CO ===')
    print('%-10s %-15s %-15s' % ('Am', 'Hien co', 'Muc tieu'))
    print('-' * 40)
    for sound in SOUNDS:
        count = count_samples(os.path.join(TEMPLATES_DIR, sound), sound)
        print('%-10s %-15d %-15d' % (sound, count, DEFAULT_COUNT))
    print('')


def main():
    parser = argparse.ArgumentParser(description='Thu mau am cho voice template')
    parser.add_argument('--sound', '-s', choices=SOUNDS,
                        help='Chi thu am dinh san (U, I, O, E, A)')
    parser.add_argument('--count', '-c', type=int, default=DEFAULT_COUNT,
                        help='So mau can thu cho moi am (mac dinh: %d)' % DEFAULT_COUNT)
    parser.add_argument('--all', '-a', action='store_true',
                        help='Hien thi so luong mau hien co')
    args = parser.parse_args()

    # Neu chi hien thi
    if args.all:
        show_all_status()
        return

    # Khoi tao microphone
    try:
        import speech_recognition as sr
    except ImportError:
        print('Loi: can cai dat speech_recognition')
        sys.exit(1)

    # Tao thu muc cho tat ca am
    for sound in SOUNDS:
        label_dir = os.path.join(TEMPLATES_DIR, sound)
        os.makedirs(label_dir, exist_ok=True)

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = False
    mic, mic_label = pick_working_microphone(sr)
    if mic is None:
        print('Khong tim thay microphone.')
        sys.exit(1)

    # Xac dinh am can thu
    target_count = args.count
    if args.sound:
        sounds_to_record = [args.sound]
    else:
        sounds_to_record = SOUNDS

    # Loc nhung am chua du target
    sounds_to_record = [s for s in sounds_to_record
                        if count_samples(os.path.join(TEMPLATES_DIR, s), s) < target_count]

    if not sounds_to_record:
        print('')
        print('Tat ca am da co day %d mau!' % target_count)
        show_all_status()
        return

    print('=' * 60)
    print('THU %d MAU AM CHO MOI NGUYEN AM' % target_count)
    print('=' * 60)
    print('Am:    U=nhay | I=phai | O=trai | E=phai | A=trai')
    print('Micro: ', mic_label)
    print('Thu muc:', TEMPLATES_DIR)
    print('=' * 60)
    print('')
    print('Se thu am: %s' % ', '.join(sounds_to_record))
    print('')

    # Dem so luong hien co
    counts = {sound: count_samples(os.path.join(TEMPLATES_DIR, sound), sound)
              for sound in sounds_to_record}

    # Thu tung am
    for sound in sounds_to_record:
        label_dir = os.path.join(TEMPLATES_DIR, sound)
        have = counts[sound]
        remaining = target_count - have

        print('')
        print('=' * 60)
        print('THU AM "%s" (%s)' % (sound, SOUND_MEANINGS[sound]))
        print('Hien co: %d mau | Can them: %d mau' % (have, remaining))
        print('=' * 60)

        idx = next_index(label_dir, sound)
        recorded = 0

        while recorded < remaining:
            result = record_one(recognizer, mic, sound, idx, target_count)
            if result == 'quit':
                print('\nDa thoat.')
                sys.exit(0)
            if result == 'ok':
                recorded += 1
                idx += 1
                counts[sound] = have + recorded
            if result == 'skip':
                idx += 1

            show_status(sounds_to_record, counts, target_count)

    # Ket qua cuoi cung
    print('')
    print('=' * 60)
    print('KET QUA CUOI CUNG:')
    print('=' * 60)
    for sound in SOUNDS:
        final_count = count_samples(os.path.join(TEMPLATES_DIR, sound), sound)
        status = '[DA DU]' if final_count >= target_count else '[CHUA DU]'
        print('  %s: %d mau %s' % (sound, final_count, status))
    print('=' * 60)
    print('')
    print('Tiep theo: python build_templates.py de cap nhat template')


if __name__ == '__main__':
    main()
