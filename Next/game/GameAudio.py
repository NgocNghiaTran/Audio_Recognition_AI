import os
import pygame as pg

_BGM_NAMES = frozenset(('overworld', 'overworld_fast'))


class Sound(object):
    def __init__(self):
        self.sounds = {}
        self.enabled = False
        self.load_sounds()

    def load_sounds(self):
        # Disable sound if no sounds folder
        sounds_dir = 'sounds'
        if not os.path.isdir(sounds_dir):
            print('[Sound] No sounds folder found, disabling audio')
            return

        sound_files = {
            'overworld': 'overworld.wav',
            'overworld_fast': 'overworld-fast.wav',
            'level_end': 'levelend.wav',
            'coin': 'coin.wav',
            'small_mario_jump': 'jump.wav',
            'big_mario_jump': 'jumpbig.wav',
            'brick_break': 'blockbreak.wav',
            'block_hit': 'blockhit.wav',
            'mushroom_appear': 'mushroomappear.wav',
            'mushroom_eat': 'mushroomeat.wav',
            'death': 'death.wav',
            'pipe': 'pipe.wav',
            'kill_mob': 'kill_mob.wav',
            'game_over': 'gameover.wav',
            'scorering': 'scorering.wav',
            'fireball': 'fireball.wav',
            'shot': 'shot.wav',
        }

        for name, filename in sound_files.items():
            filepath = os.path.join(sounds_dir, filename)
            if os.path.isfile(filepath):
                self.sounds[name] = pg.mixer.Sound(filepath)
            else:
                print(f'[Sound] Missing: {filepath}')

        self.enabled = len(self.sounds) > 0
        if not self.enabled:
            print('[Sound] No sound files found, disabling audio')

    def play(self, name, loops, volume):
        if not self.enabled or name not in _BGM_NAMES and name not in self.sounds:
            return
        if name in _BGM_NAMES:
            return
        self.sounds[name].play(loops=loops)
        self.sounds[name].set_volume(volume)

    def stop(self, name):
        if not self.enabled or name not in self.sounds:
            return
        self.sounds[name].stop()

    def start_fast_music(self, core):
        pass