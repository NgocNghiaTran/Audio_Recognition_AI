from sound.config import TEMPLATES_PATH
from sound.matching.matcher import VoiceMfccMatcher


def main():
    matcher = VoiceMfccMatcher()
    if matcher.enabled:
        print('OK:', TEMPLATES_PATH)
    else:
        print('That bai. Kiem tra sound/data/templates/ va pip install librosa')


if __name__ == '__main__':
    main()
