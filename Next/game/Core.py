from os import environ

import pygame as pg
from pygame.locals import *

from game.Const import *
from game.Map import Map
from game.MenuManager import MenuManager
from game.GameAudio import Sound
from sound.pipeline.voice_input_pipeline import VoiceInputPipeline
from sound.testing.guided_test import VoiceGuidedTest


class Core(object):
    """

    Main class.

    """
    def __init__(self):
        environ['SDL_VIDEO_CENTERED'] = '1'
        pg.mixer.pre_init(44100, -16, 2, 1024)
        pg.init()
        pg.display.set_caption('Mario by Nghia')
        pg.display.set_mode((WINDOW_W, WINDOW_H))

        self.screen = pg.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pg.time.Clock()

        self.oWorld = Map('1-1')
        self.oSound = Sound()
        self.oMM = MenuManager(self)

        self.run = True
        self.keyR = False
        self.keyL = False
        self.keyU = False
        self.keyD = False
        self.keyShift = False
        self.keyR_keyboard = False
        self.keyL_keyboard = False
        self.keyU_keyboard = False
        self.keyR_voice_until = 0
        self.keyL_voice_until = 0
        self.keyU_voice_until = 0

        self.voice_test = VoiceGuidedTest(get_chunk_id=lambda: self.voice.get_chunk_id())
        self.voice = VoiceInputPipeline(get_expected=self.voice_test.get_expected)
        self.voice.start()

    def main_loop(self):
        try:
            while self.run:
                self.input()
                self.update()
                self.render()
                self.clock.tick(FPS)
        finally:
            self.voice.stop()
            pg.quit()

    def input(self):
        if self.get_mm().currentGameState == 'Game':
            self.input_player()
        else:
            self.input_menu()

    def input_player(self):
        for e in pg.event.get():

            if e.type == pg.QUIT:
                self.run = False

            elif e.type == KEYDOWN:
                if e.key == K_RIGHT:
                    self.keyR_keyboard = True
                elif e.key == K_LEFT:
                    self.keyL_keyboard = True
                elif e.key == K_DOWN:
                    self.keyD = True
                elif e.key == K_UP:
                    self.keyU_keyboard = True
                elif e.key == K_LSHIFT:
                    self.keyShift = True
                elif e.key == K_t:
                    self.voice.reset_chunk_counter()
                    self.voice_test.start()

            elif e.type == KEYUP:
                if e.key == K_RIGHT:
                    self.keyR_keyboard = False
                elif e.key == K_LEFT:
                    self.keyL_keyboard = False
                elif e.key == K_DOWN:
                    self.keyD = False
                elif e.key == K_UP:
                    self.keyU_keyboard = False
                elif e.key == K_LSHIFT:
                    self.keyShift = False

    def input_menu(self):
        for e in pg.event.get():
            if e.type == pg.QUIT:
                self.run = False

            elif e.type == KEYDOWN:
                if e.key == K_RETURN:
                    self.get_mm().start_loading()

    def update(self):
        self.voice_test.update()
        self.process_voice_events()
        self.sync_jump_key()
        self.get_mm().update(self)

    def render(self):
        self.get_mm().render(self)
        if self.get_mm().currentGameState == 'Game':
            self.voice_test.render(self.screen)

    def get_map(self):
        return self.oWorld

    def get_mm(self):
        return self.oMM

    def get_sound(self):
        return self.oSound

    def process_voice_events(self):
        for event_type, payload in self.voice.poll_events():
            if event_type == 'TRANSCRIPT':
                text = payload['text'] if isinstance(payload, dict) else payload
                chunk_id = payload.get('chunk_id', '?') if isinstance(payload, dict) else '?'
                print(f'[VOICE] chunk={chunk_id} transcript="{text}"')
            elif event_type == 'NO_LISTEN':
                print(f'[VOICE] no listen ({payload})')
            elif event_type == 'JUMP' and self.get_mm().currentGameState == 'Game':
                now = pg.time.get_ticks()
                self.keyU_voice_until = max(self.keyU_voice_until, now + 120)
            elif event_type == 'RIGHT' and self.get_mm().currentGameState == 'Game':
                now = pg.time.get_ticks()
                self.keyR_voice_until = max(self.keyR_voice_until, now + 220)
                self.keyL_voice_until = min(self.keyL_voice_until, now)
            elif event_type == 'LEFT' and self.get_mm().currentGameState == 'Game':
                now = pg.time.get_ticks()
                self.keyL_voice_until = max(self.keyL_voice_until, now + 220)
                self.keyR_voice_until = min(self.keyR_voice_until, now)

    def sync_jump_key(self):
        now = pg.time.get_ticks()
        self.keyU = self.keyU_keyboard or (now <= self.keyU_voice_until)
        self.keyR = self.keyR_keyboard or (now <= self.keyR_voice_until)
        self.keyL = self.keyL_keyboard or (now <= self.keyL_voice_until)
