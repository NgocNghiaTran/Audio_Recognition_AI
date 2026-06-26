"""HCI audio pipeline: capture -> preprocess -> features -> matching -> game."""
from sound.matching.matcher import VoiceMfccMatcher
from sound.pipeline.input import VoiceInput
from sound.pipeline.voice_input_pipeline import VoiceInputPipeline

__all__ = ('VoiceInput', 'VoiceInputPipeline', 'VoiceMfccMatcher', 'VoiceGuidedTest')


def __getattr__(name):
    if name == 'VoiceGuidedTest':
        from sound.testing.guided_test import VoiceGuidedTest
        return VoiceGuidedTest
    raise AttributeError(name)
