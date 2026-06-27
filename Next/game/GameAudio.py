import pygame as pg

_BGM_NAMES = frozenset(('overworld', 'overworld_fast'))


class Sound(object):
    def __init__(self):
        self.sounds = {}

    def load_sounds(self):
        pass  # Khong tai am thanh

    def play(self, name, loops, volume):
        pass  # Khong phat am thanh

    def stop(self, name):
        pass

    def start_fast_music(self, core):
        pass