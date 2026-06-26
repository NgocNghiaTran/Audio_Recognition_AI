import pygame as pg

from game.Const import WINDOW_W
from sound.config import CHUNKS_PER_PHASE, guided_test_phases

_PHASES = tuple(
    (label, command, CHUNKS_PER_PHASE)
    for label, command, _start, _end in guided_test_phases()
)


class VoiceGuidedTest(object):
    def __init__(self, get_chunk_id=None):
        self._get_chunk_id = get_chunk_id or (lambda: 0)
        self.active = False
        self.finished = False
        self._font_lg = None
        self._font_sm = None

    def start(self):
        self.active = True
        self.finished = False
        parts = []
        for label, command, start, end in guided_test_phases():
            parts.append('%s(%d-%d)=%s' % (label, start, end, command))
        print('[VOICE TEST] Bat dau 5 phase x %d chunk: %s' % (
            CHUNKS_PER_PHASE, ' -> '.join(parts)))

    def stop(self):
        self.active = False
        self.finished = True

    def _ensure_fonts(self):
        if self._font_lg is None:
            self._font_lg = pg.font.Font('fonts/emulogic.ttf', 28)
            self._font_sm = pg.font.Font('fonts/emulogic.ttf', 18)

    def _phase_info(self):
        chunk = self._get_chunk_id()
        if chunk < 1:
            chunk = 1
        offset = 0
        for idx, (label, command, n_chunks) in enumerate(_PHASES):
            phase_end = offset + n_chunks
            if chunk <= phase_end:
                return {
                    'label': label, 'command': command,
                    'phase_no': idx + 1, 'phase_total': len(_PHASES),
                    'chunk_id': chunk, 'remaining_chunks': phase_end - chunk,
                }
            offset += n_chunks
        return None

    def update(self):
        if not self.active:
            return
        if self._phase_info() is None:
            self.active = False
            self.finished = True
            print('[VOICE TEST] Ket thuc (%d chunk). Xem Excel voice_test_*_MFCC.' % (
                CHUNKS_PER_PHASE * len(_PHASES)))

    def get_expected(self):
        info = self._phase_info()
        if info and self.active:
            return info['label'], info['command']
        return '', ''

    def render(self, screen):
        if not self.active:
            if self.finished:
                self._ensure_fonts()
                txt = self._font_sm.render('VOICE TEST XONG - xem Excel Voice Log', False, (255, 220, 80))
                screen.blit(txt, (10, 60))
            return
        info = self._phase_info()
        if not info:
            return
        self._ensure_fonts()
        overlay = pg.Surface((WINDOW_W, 90), pg.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        screen.blit(overlay, (0, 50))
        lines = [
            'VOICE TEST  Phien %d/%d  Chunk %d  Con %d' % (
                info['phase_no'], info['phase_total'], info['chunk_id'], info['remaining_chunks']),
            'Hay doc am:  %s' % info['label'],
            'Mong doi lenh: %s  (T=bat dau lai)' % info['command'],
        ]
        y = 58
        for i, line in enumerate(lines):
            color = (255, 255, 80) if i == 1 else (255, 255, 255)
            surf = self._font_lg.render(line, False, color) if i == 1 else self._font_sm.render(line, False, color)
            screen.blit(surf, (12, y))
            y += 28 if i == 1 else 22
