"""VoiceInput với Multi-Buffer Pipeline - đúng như đề tài mô tả.

Kiến trúc:
- Buffer 1: đang ghi (recording)
- Buffer 2: đang chờ (waiting)
- Buffer 3: đang xử lý (processing)
- Multiple workers xử lý song song

Điểm khác biệt với VoiceInput cũ:
- Dùng TripleBuffer thay vì single queue
- Nhiều worker xử lý đồng thời
- Overlap giữa thu và xử lý
"""
import os
import queue
import re
import threading
import time
from typing import Optional, Tuple

import numpy as np

from sound.capture.buffer import TripleBuffer, AudioChunk
from sound.capture.gate import voice_is_loud_enough
from sound.capture.mic import pick_working_microphone
from sound.config import SAMPLE_RATE
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


class VoiceInputPipeline:
    """Voice input với multi-buffer pipeline.

    Kiến trúc theo đề tài:
    - Producer thread: liên tục thu âm vào triple buffer
    - Worker pool: xử lý song song (MFCC hoặc STT)
    - Main thread: nhận kết quả và emit events

    Điểm mạnh:
    - 3 buffers xoay vòng: recording → waiting → queue
    - Multiple workers xử lý đồng thời
    - Overlap giữa thu âm và xử lý → giảm latency
    """

    def __init__(self, jump_keywords=None, language='en-US', cooldown_ms=650,
                 stt_backend='auto', whisper_model='tiny.en', get_expected=None,
                 num_workers: int = 2, chunk_duration: float = 0.30):
        self._get_expected = get_expected or (lambda: ('', ''))

        # Keywords
        ex = {w.strip().lower() for w in (jump_keywords or []) if isinstance(w, str) and w.strip()}
        self._jump_words = _DEFAULT_JUMP_WORDS | _STT_ALIASES_JUMP | ex
        self._left_words = _DEFAULT_LEFT_WORDS | _STT_ALIASES_LEFT
        self._right_words = _DEFAULT_RIGHT_WORDS | _STT_ALIASES_RIGHT

        self.language = language
        self.whisper_model = whisper_model
        self.cooldown_ms = cooldown_ms
        self.num_workers = num_workers
        self.chunk_duration = chunk_duration

        # Energy threshold (tăng để tránh noise trigger)
        self.energy_threshold = int(os.environ.get('MARIO_VOICE_ENERGY', '300') or 300)
        self.noise_floor_rms = 120

        # MFCC matcher
        self.mfcc = VoiceMfccMatcher(use_margin=False)
        self.mfcc_cooldown_ms = 600  # Cooldown: 0.6s (>= 2x chunk_duration)
        self.last_mfcc_ms = 0
        self.last_cmd_time = {}  # Cooldown riêng cho từng command

        # Determine mode
        if self.mfcc.enabled:
            self.recognition_mode = 'mfcc'
            self.chunk_duration = 0.50
        else:
            self.recognition_mode = 'stt'

        # Triple buffer config
        self.chunk_samples = int(SAMPLE_RATE * self.chunk_duration)  # VD: 16000 * 0.5 = 8000 samples

        # Triple buffer
        self._triple_buffer: Optional[TripleBuffer] = None
        self._sr = self._recognizer = self._mic = None
        self._mic_label = ''

        # Events queue (output)
        self.events = queue.Queue()

        # Metrics
        self.chunk_counter = 0
        self.last_jump_ms = 0
        self.last_no_listen_ms = 0
        self.no_listen_cooldown_ms = 5000
        self.metrics = {
            'chunks_recorded': 0,
            'chunks_processed': 0,
            'chunks_dropped': 0,
            'commands_emitted': 0,
            'capture_to_action_ms': [],
            'buffer_queue_sizes': [],  # Track queue size over time
            'worker_utilization': [],  # Track worker busy/idle ratio
        }

        # STT config
        self.enabled = True
        self._whisper = None
        self.stt_mode = 'none'
        self._use_faster_whisper = self._use_google = False
        self._resolve_stt_backend(stt_backend)
        self._setup_engine()

        # Data logger
        self._data_logger = VoiceDataLogger(
            stt_backend=self.stt_mode,
            test_mode='MFCC_PIPELINE' if self.mfcc.enabled else 'STT_PIPELINE',
        )

        # Control
        self.running = False
        self._threads: list = []

    def _resolve_stt_backend(self, stt_backend):
        if stt_backend in ('auto', 'faster_whisper'):
            try:
                import faster_whisper
                self._use_faster_whisper = True
                self.stt_mode = 'faster_whisper'
            except ImportError:
                if stt_backend == 'faster_whisper':
                    print('[VOICE] faster-whisper not installed')
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
                raise OSError('No microphone')
            print('[VOICE-PIPELINE] Microphone:', self._mic_label)
        except Exception as exc:
            self.enabled = False
            print('[VOICE-PIPELINE] Disabled:', exc)

    def start(self):
        """Khởi động pipeline: producer + workers."""
        if not self.enabled or self.running:
            return

        self.running = True

        # Khởi tạo triple buffer
        self._triple_buffer = TripleBuffer(
            chunk_samples=self.chunk_samples,
            num_workers=self.num_workers
        )

        # Set callback xử lý
        self._triple_buffer.set_process_callback(self._process_chunk)

        # Producer: thu âm liên tục
        producer_thread = threading.Thread(
            target=self._producer_loop,
            daemon=True,
            name='VoicePipeline-Producer'
        )
        producer_thread.start()
        self._threads.append(producer_thread)

        # Workers: xử lý song song
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True,
                name=f'VoicePipeline-Worker-{i}'
            )
            t.start()
            self._threads.append(t)

        print(f'[VOICE-PIPELINE] Started (mode={self.recognition_mode}, workers={self.num_workers}, chunk={self.chunk_duration}s)')
        print('[VOICE-PIPELINE] Level 1: U=jump | I,E=right | O,A=left')

    def stop(self):
        if not self.running:
            return

        self.running = False

        # Stop triple buffer
        if self._triple_buffer:
            self._triple_buffer.stop()

        # Chờ threads kết thúc
        for t in self._threads:
            t.join(timeout=1.0)

        self._print_metrics_summary()
        self._data_logger.log_summary(self.metrics)
        self._data_logger.close()

        print('[VOICE-PIPELINE] Stopped')

    def _producer_loop(self):
        """Producer: liên tục thu âm và ghi vào triple buffer."""
        sr = self._sr
        recognizer = self._recognizer
        mic = self._mic

        # Calibrate noise floor
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.4)
                try:
                    self.noise_floor_rms = int(recognizer.energy_threshold)
                except Exception:
                    self.noise_floor_rms = 120
        except Exception as exc:
            print('[VOICE-PIPELINE] Mic setup failed:', exc)
            self.running = False
            return

        # Thu liên tục
        while self.running:
            try:
                with mic as source:
                    # Thu 1 chunk = chunk_duration giây
                    audio = recognizer.record(source, duration=self.chunk_duration)

                captured_at_ms = int(time.time() * 1000)

                # Check loudness
                loud, rms, self.noise_floor_rms = voice_is_loud_enough(
                    audio, self.energy_threshold, self.noise_floor_rms
                )

                # Chuyển sang numpy từ raw_data
                raw = audio.get_raw_data(convert_rate=SAMPLE_RATE, convert_width=2)
                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

                # Tạo chunk
                chunk = AudioChunk(
                    pcm=pcm,
                    sample_rate=SAMPLE_RATE,
                    captured_at_ms=captured_at_ms,
                    chunk_id=self.chunk_counter,
                    is_loud=loud,
                    rms=rms,
                )
                self.chunk_counter += 1
                self.metrics['chunks_recorded'] += 1

                # Xử lý chunk (MFCC rất nhanh, xử lý luôn trong producer)
                self._process_chunk(chunk)
                self.metrics['chunks_processed'] += 1

                # Track queue size
                if self._triple_buffer:
                    self.metrics['buffer_queue_sizes'].append(
                        self._triple_buffer._process_queue.qsize()
                    )

            except Exception as e:
                print(f'[VOICE-PIPELINE] Producer error: {e}')
                continue

    def _worker_loop(self, worker_id: int):
        """Worker: xử lý chunks từ triple buffer."""
        while self.running:
            try:
                if self._triple_buffer:
                    # Lấy chunk từ buffer
                    chunk = self._triple_buffer._process_queue.get(timeout=0.1)
                    self.metrics['chunks_processed'] += 1

                    # Xử lý
                    self._process_chunk(chunk)

            except queue.Empty:
                # Worker idle
                self.metrics['worker_utilization'].append(0)
                continue
            except Exception as e:
                print(f'[VOICE-PIPELINE] Worker {worker_id} error: {e}')

    def _process_chunk(self, chunk: AudioChunk):
        """Xử lý một chunk: MFCC hoặc STT."""
        if not chunk.is_loud:
            self._log_event(
                text='-', command='-', chunk_id=chunk.chunk_id, note='too_quiet'
            )
            return

        # MFCC mode: classify trực tiếp
        if self.mfcc.enabled:
            self._try_mfcc_command(chunk)
        else:
            # STT mode: transcribe
            self._do_stt(chunk)

    def _try_mfcc_command(self, chunk):
        """Xử lý MFCC command."""
        now = int(time.time() * 1000)
        if now - self.last_mfcc_ms < self.mfcc_cooldown_ms:
            return

        label, score, scores = self.mfcc.classify_audio_from_pcm(chunk.pcm)
        if not label:
            return

        command = self.mfcc.command_for_label(label)
        if not command:
            return

        # Per-command cooldown (ngăn cùng command spam)
        cmd_key = f'mfcc_{command}'
        last_time = self.last_cmd_time.get(cmd_key, 0)
        if now - last_time < self.mfcc_cooldown_ms:
            return

        self.last_mfcc_ms = now
        self.last_cmd_time[cmd_key] = now
        payload = f'mfcc:{label}({score:.2f})'
        latency_ms = self._emit_command(command, payload, chunk.captured_at_ms, chunk.chunk_id)
        self._log_event(
            text=payload, command=command, chunk_id=chunk.chunk_id,
            action_latency_ms=latency_ms, note='mfcc_ok'
        )

    def _do_stt(self, chunk):
        """Xử lý STT."""
        now_ms = int(time.time() * 1000)
        latency_ms = now_ms - chunk.captured_at_ms

        try:
            if self._use_faster_whisper:
                text = self._get_whisper().transcribe_from_pcm(chunk.pcm)
            else:
                # Convert PCM to AudioData
                import speech_recognition as sr
                raw_data = (chunk.pcm * 32767).astype(np.int16).tobytes()
                audio_data = sr.AudioData(raw_data, SAMPLE_RATE, 2)
                text = self._recognizer.recognize_google(audio_data, language=self.language)

            text = (text or '').strip()
            if not text:
                self._log_event(
                    text='-', command='-', chunk_id=chunk.chunk_id,
                    note='stt_empty'
                )
                return

            # Emit transcript
            self.events.put(('TRANSCRIPT', {
                'text': text,
                'chunk_id': chunk.chunk_id,
                'latency_ms': latency_ms,
            }))

            # Check commands
            command = ''
            action_latency_ms = ''

            if self._has_jump_command(text):
                command = 'JUMP'
                action_latency_ms = self._emit_command(
                    'JUMP', text, chunk.captured_at_ms, chunk.chunk_id
                )

            direction = self._last_direction_command(text)
            if direction:
                command = direction
                action_latency_ms = self._emit_command(
                    direction, text, chunk.captured_at_ms, chunk.chunk_id
                )

            self._log_event(
                text=text, command=command or '-', chunk_id=chunk.chunk_id,
                stt_latency_ms=latency_ms, action_latency_ms=action_latency_ms,
                note='ok' if command else 'no_command'
            )

        except Exception as e:
            self._log_event(chunk_id=chunk.chunk_id, note=f'stt_error:{e}')

    def _has_jump_command(self, text):
        return any(w in self._tokenize(text) for w in self._jump_words)

    def _last_direction_command(self, text):
        last_direction = None
        for w in self._tokenize(text):
            if w in self._right_words:
                last_direction = 'RIGHT'
            elif w in self._left_words:
                last_direction = 'LEFT'
        return last_direction

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[a-zA-Z0-9']+", (text or '').lower())

    def _get_whisper(self):
        if self._whisper is None:
            self._whisper = WhisperTranscriber(self.whisper_model)
        return self._whisper

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
            'text': text,
            'chunk_id': chunk_id,
            'latency_ms': latency_ms,
        }))

        print(f'[VOICE-PIPELINE] {event_type} chunk={chunk_id} latency={latency_ms}ms text="{text}"')
        return latency_ms

    def _log_event(self, **kwargs):
        exp_label, exp_cmd = self._get_expected()
        self._data_logger.log_voice_event(
            expected_label=exp_label or '',
            expected_command=exp_cmd or '',
            **kwargs
        )

    def _print_metrics_summary(self):
        m = self.metrics
        print('[VOICE-PIPELINE] === Pipeline Metrics ===')
        print(f'  Chunks recorded: {m["chunks_recorded"]}')
        print(f'  Chunks processed: {m["chunks_processed"]}')
        print(f'  Chunks dropped: {m["chunks_dropped"]}')
        print(f'  Commands emitted: {m["commands_emitted"]}')

        if m['capture_to_action_ms']:
            avg_ms = sum(m['capture_to_action_ms']) / len(m['capture_to_action_ms'])
            print(f'  Capture→action avg: {avg_ms:.0f}ms (n={len(m["capture_to_action_ms"])})')

        if m['buffer_queue_sizes']:
            avg_q = sum(m['buffer_queue_sizes']) / len(m['buffer_queue_sizes'])
            print(f'  Avg queue size: {avg_q:.1f}')

        if m['worker_utilization']:
            busy = sum(1 for x in m['worker_utilization'] if x > 0)
            total = len(m['worker_utilization'])
            print(f'  Worker utilization: {busy}/{total} ({100*busy/total:.0f}%)')

    def poll_events(self):
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def get_chunk_id(self):
        return self.chunk_counter

    def reset_chunk_counter(self):
        self.chunk_counter = 0
