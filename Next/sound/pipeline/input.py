import os
import queue
import re
import threading
import time

from sound.capture.gate import voice_is_loud_enough
from sound.capture.mic import pick_working_microphone
from sound.logging.data_logger import VoiceDataLogger
from sound.matching.matcher import VoiceMfccMatcher
from sound.stt.keywords import (
    _DEFAULT_JUMP_WORDS,
    _DEFAULT_LEFT_WORDS,
    _DEFAULT_RIGHT_WORDS,
    _STT_ALIASES_JUMP,
    _STT_ALIASES_LEFT,
    _STT_ALIASES_RIGHT,
)
from sound.stt.whisper import WhisperTranscriber


class VoiceInput(object):
    def __init__(self, jump_keywords=None, language='en-US', cooldown_ms=650,
                 stt_backend='auto', whisper_model='tiny.en', get_expected=None):
        self._get_expected = get_expected or (lambda: ('', ''))
        ex = {w.strip().lower() for w in (jump_keywords or []) if isinstance(w, str) and w.strip()}
        self._jump_words = _DEFAULT_JUMP_WORDS | _STT_ALIASES_JUMP | ex
        self._left_words = _DEFAULT_LEFT_WORDS | _STT_ALIASES_LEFT
        self._right_words = _DEFAULT_RIGHT_WORDS | _STT_ALIASES_RIGHT
        self.language = language
        self.whisper_model = whisper_model
        self.cooldown_ms = cooldown_ms
        self.mfcc_cooldown_ms = 250
        self.last_mfcc_ms = 0
        self.energy_threshold = int(os.environ.get('MARIO_VOICE_ENERGY', '150') or 150)
        self.chunk_duration = 0.30
        self.sample_rate = 16000
        self.mfcc = VoiceMfccMatcher(use_margin=False)
        if self.mfcc.enabled:
            self.chunk_duration = 0.50
            self.recognition_mode = 'mfcc'
        else:
            self.recognition_mode = 'stt'

        self.events = queue.Queue()
        self.stt_queue = queue.Queue(maxsize=3)
        self.running = False
        self.thread = None
        self.stt_thread = None
        self.last_jump_ms = 0
        self.last_no_listen_ms = 0
        self.no_listen_cooldown_ms = 5000
        self.noise_floor_rms = 120
        self.chunk_counter = 0
        self.metrics = {
            'chunks_captured': 0,
            'chunks_queued': 0,
            'chunks_dropped': 0,
            'stt_done': 0,
            'commands_emitted': 0,
            'capture_to_stt_ms': [],
            'capture_to_action_ms': [],
        }

        self.enabled = True
        self._sr = self._recognizer = self._mic = None
        self._mic_label = ''
        self._whisper = None
        self.stt_mode = 'none'
        self._use_faster_whisper = self._use_google = False
        self._resolve_stt_backend(stt_backend)
        self._setup_engine()
        self._data_logger = VoiceDataLogger(
            stt_backend=self.stt_mode,
            test_mode='MFCC' if self.mfcc.enabled else 'STT',
        )

    def _resolve_stt_backend(self, stt_backend):
        if stt_backend in ('auto', 'faster_whisper'):
            try:
                import faster_whisper  # noqa: F401
                self._use_faster_whisper = True
                self.stt_mode = 'faster_whisper'
            except ImportError:
                if stt_backend == 'faster_whisper':
                    print('[VOICE] faster-whisper not installed. pip install faster-whisper')
        if not self._use_faster_whisper and stt_backend in ('auto', 'google', 'faster_whisper'):
            self._use_google = True
            self.stt_mode = 'google'
        if self.stt_mode == 'none':
            self._use_google = True
            self.stt_mode = 'google'

    def _setup_engine(self):
        try:
            import speech_recognition as sr
            self._sr = sr
            r = sr.Recognizer()
            r.dynamic_energy_threshold = False
            r.pause_threshold = 0.15
            r.non_speaking_duration = 0.1
            self._recognizer = r
            self._mic, self._mic_label = pick_working_microphone(sr)
            if self._mic is None:
                raise OSError('No Default Input Device Available')
            print('[VOICE] Microphone:', self._mic_label)
        except Exception as exc:
            self.enabled = False
            self._mic = None
            self._mic_label = ''
            print('[VOICE] Disabled:', exc)
            print('[VOICE] Install: pip install SpeechRecognition pyaudio')

    def start(self):
        if not self.enabled or self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.stt_thread = threading.Thread(target=self._stt_loop, daemon=True)
        self.thread.start()
        self.stt_thread.start()
        print('[VOICE] Started (mode: %s, STT: %s, chunk: %.0fms)' % (
            self.recognition_mode, self.stt_mode, self.chunk_duration * 1000))
        print('[VOICE] Level 1: U=jump | I,E=right | O,A=left')

    def stop(self):
        if not self.running:
            return
        self.running = False
        for t in (self.thread, self.stt_thread):
            if t is not None:
                t.join(timeout=1.0)
        self._print_metrics_summary()
        self._data_logger.log_summary(self.metrics)
        self._data_logger.close()
        print('[VOICE] Stopped')

    @staticmethod
    def _avg(values):
        return int(sum(values) / len(values)) if values else 0

    def _print_metrics_summary(self):
        m = self.metrics
        print('[VOICE] --- Baseline metrics ---')
        print('[VOICE] chunks captured=%d queued=%d dropped=%d stt_done=%d commands=%d' % (
            m['chunks_captured'], m['chunks_queued'], m['chunks_dropped'],
            m['stt_done'], m['commands_emitted']))
        if m['capture_to_stt_ms']:
            print('[VOICE] capture->STT avg=%dms (n=%d)' % (
                self._avg(m['capture_to_stt_ms']), len(m['capture_to_stt_ms'])))
        if m['capture_to_action_ms']:
            print('[VOICE] capture->action avg=%dms (n=%d)' % (
                self._avg(m['capture_to_action_ms']), len(m['capture_to_action_ms'])))

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[a-zA-Z0-9']+", (text or '').lower())

    def _has_jump_command(self, text):
        return any(w in self._jump_words for w in self._tokenize(text))

    def _last_direction_command(self, text):
        last_direction = None
        for w in self._tokenize(text):
            if w in self._right_words:
                last_direction = 'RIGHT'
            elif w in self._left_words:
                last_direction = 'LEFT'
        return last_direction

    def _log_event(self, **kwargs):
        exp_label, exp_cmd = self._get_expected()
        self._data_logger.log_voice_event(
            expected_label=exp_label or '',
            expected_command=exp_cmd or '',
            **kwargs)

    def get_chunk_id(self):
        return self.chunk_counter

    def reset_chunk_counter(self):
        self.chunk_counter = 0

    def poll_events(self):
        out, q = [], self.events
        while True:
            try:
                out.append(q.get_nowait())
            except queue.Empty:
                return out

    def _listen_loop(self):
        sr, recognizer, mic = self._sr, self._recognizer, self._mic
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.4)
                try:
                    self.noise_floor_rms = int(recognizer.energy_threshold)
                except Exception:
                    self.noise_floor_rms = 120
        except Exception as exc:
            print('[VOICE] Mic setup failed:', exc)
            self.running = False
            return

        while self.running:
            try:
                with mic as source:
                    audio = recognizer.record(source, duration=self.chunk_duration)
                captured_at_ms = int(time.time() * 1000)
                self.chunk_counter += 1
                self.metrics['chunks_captured'] += 1
                loud, _, self.noise_floor_rms = voice_is_loud_enough(
                    audio, self.energy_threshold, self.noise_floor_rms)
                if not loud:
                    self._log_event(
                        text='-', command='-', chunk_id=self.chunk_counter, note='too_quiet')
                    continue
                if self.mfcc.enabled:
                    self._try_mfcc_command(audio, captured_at_ms, self.chunk_counter)
                chunk = (audio, captured_at_ms, self.chunk_counter)
                try:
                    self.stt_queue.put_nowait(chunk)
                    self.metrics['chunks_queued'] += 1
                except queue.Full:
                    self.metrics['chunks_dropped'] += 1
                    self._log_event(
                        text='-', command='-', chunk_id=self.chunk_counter, note='queue_full')
            except Exception as exc:
                print('[VOICE] Runtime error:', exc)
                time.sleep(0.2)

    def _try_mfcc_command(self, audio, captured_at_ms, chunk_id):
        now = int(time.time() * 1000)
        if now - self.last_mfcc_ms < self.mfcc_cooldown_ms:
            return
        label, score, scores = self.mfcc.classify_audio(audio)
        if not label:
            reject = scores.get('_reject', '')
            if reject and score >= 0.60:
                score_txt = ','.join(
                    '%s:%.2f' % (k, scores[k]) for k in sorted(scores) if not k.startswith('_'))
                self._log_event(
                    text='mfcc:reject(%.2f)' % score, command='-',
                    chunk_id=chunk_id, note='mfcc_reject|%s|%s' % (reject, score_txt))
            return
        command = self.mfcc.command_for_label(label)
        if not command:
            return
        self.last_mfcc_ms = now
        score_txt = ','.join(
            '%s:%.2f' % (k, scores[k]) for k in sorted(scores) if not k.startswith('_'))
        payload = 'mfcc:%s(%.2f)' % (label, score)
        action_latency_ms = self._emit_command(command, payload, captured_at_ms, chunk_id)
        self._log_event(
            text=payload, command=command, chunk_id=chunk_id,
            action_latency_ms=action_latency_ms, note='mfcc|%s' % score_txt)

    def _get_whisper(self):
        if self._whisper is None:
            self._whisper = WhisperTranscriber(self.whisper_model)
        return self._whisper

    def _stt_loop(self):
        sr, recognizer = self._sr, self._recognizer
        while self.running:
            try:
                audio, captured_at_ms, chunk_id = self.stt_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                if self._use_faster_whisper:
                    text = self._get_whisper().transcribe(audio)
                else:
                    text = recognizer.recognize_google(audio, language=self.language)
                text = (text or '').strip()
                stt_done_ms = int(time.time() * 1000)
                self.metrics['stt_done'] += 1
                self.metrics['capture_to_stt_ms'].append(stt_done_ms - captured_at_ms)
                stt_latency_ms = stt_done_ms - captured_at_ms
                if not text:
                    self._log_event(
                        text='-', command='-', chunk_id=chunk_id,
                        stt_latency_ms=stt_latency_ms, note='stt_empty')
                    self._emit_no_listen('unknown', chunk_id=chunk_id)
                    continue
                self.events.put(('TRANSCRIPT', {'text': text, 'chunk_id': chunk_id}))
                if self.mfcc.enabled:
                    self._log_event(
                        text=text, command='-', chunk_id=chunk_id,
                        stt_latency_ms=stt_latency_ms, note='stt_log_only')
                    continue
                command = ''
                action_latency_ms = ''
                if self._has_jump_command(text):
                    command = 'JUMP'
                    action_latency_ms = self._emit_command(
                        'JUMP', text, captured_at_ms, chunk_id)
                direction = self._last_direction_command(text)
                if direction is not None:
                    command = direction
                    action_latency_ms = self._emit_command(
                        direction, text, captured_at_ms, chunk_id)
                self._log_event(
                    text=text, command=command or '-', chunk_id=chunk_id,
                    stt_latency_ms=stt_latency_ms, action_latency_ms=action_latency_ms,
                    note='ok' if command else 'no_command')
            except sr.UnknownValueError:
                self._log_event(chunk_id=chunk_id, note='stt_unknown')
                self._emit_no_listen('unknown', chunk_id=chunk_id)
            except sr.RequestError:
                self._log_event(chunk_id=chunk_id, note='stt_request_error')
                self._emit_no_listen('request_error', chunk_id=chunk_id)
            except Exception:
                self._log_event(chunk_id=chunk_id, note='stt_error')
                self._emit_no_listen('stt_error', chunk_id=chunk_id)

    def _emit_command(self, event_type, text, captured_at_ms, chunk_id):
        now = int(time.time() * 1000)
        if event_type == 'JUMP' and now - self.last_jump_ms < self.cooldown_ms:
            return ''
        if event_type == 'JUMP':
            self.last_jump_ms = now
        latency_ms = now - captured_at_ms
        self.metrics['commands_emitted'] += 1
        self.metrics['capture_to_action_ms'].append(latency_ms)
        self.events.put((event_type, {
            'text': text, 'chunk_id': chunk_id, 'latency_ms': latency_ms}))
        print('[VOICE] %s chunk=%d latency=%dms text="%s"' % (
            event_type, chunk_id, latency_ms, text))
        return latency_ms

    def _emit_no_listen(self, reason, chunk_id=0):
        now = int(time.time() * 1000)
        if now - self.last_no_listen_ms >= self.no_listen_cooldown_ms:
            self.last_no_listen_ms = now
            self.events.put(('NO_LISTEN', reason))
            self._log_event(text='-', command='-', chunk_id=chunk_id, note=reason)
