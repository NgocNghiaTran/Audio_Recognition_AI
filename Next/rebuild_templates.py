"""Rebuild templates sau khi thu them mau A."""
from sound.config import TEMPLATES_DIR, TEMPLATES_PATH
from sound.matching.matcher import VoiceMfccMatcher

import os
import sys


def main():
    print('=' * 50)
    print('REBUILD TEMPLATES')
    print('=' * 50)
    print()

    # Dem mau hien tai
    for label in ['U', 'I', 'O', 'E', 'A']:
        label_dir = os.path.join(TEMPLATES_DIR, label)
        if os.path.isdir(label_dir):
            count = len([f for f in os.listdir(label_dir) if f.endswith('.wav')])
            print('  %s: %d mau' % (label, count))

    print()

    # Xoa template cu
    if os.path.exists(TEMPLATES_PATH):
        print('Xoa template cu:', TEMPLATES_PATH)
        os.remove(TEMPLATES_PATH)

    print()
    print('Dang build templates moi...')
    print()

    # Build templates moi
    matcher = VoiceMfccMatcher()

    if matcher.enabled:
        print()
        print('=' * 50)
        print('BUILD THANH CONG!')
        print('=' * 50)
        print()
        print('Template moi:', TEMPLATES_PATH)
        return True
    else:
        print()
        print('=' * 50)
        print('BUILD THAT BAI!')
        print('=' * 50)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
