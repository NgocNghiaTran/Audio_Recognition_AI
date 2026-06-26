"""Ring Buffer + Triple Buffer cho audio streaming.

Thiết kế theo đề tài:
- Buffer 1: đang ghi (recording)
- Buffer 2: đang chờ (waiting)
- Buffer 3: đang xử lý (processing)
- Nhiều worker xử lý song song

Tại một thời điểm, cả 3 buffer đều active và có thể overlap.
"""
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any

import numpy as np


@dataclass
class AudioChunk:
    """Một đoạn audio đã thu."""
    pcm: np.ndarray  # Float32 array, normalized [-1, 1]
    sample_rate: int
    captured_at_ms: int  # Timestamp khi bắt đầu thu
    chunk_id: int
    is_loud: bool = True
    rms: float = 0.0

    @property
    def duration_ms(self) -> int:
        return int(len(self.pcm) / self.sample_rate * 1000)


@dataclass
class BufferStats:
    """Thống kê buffer."""
    total_recorded: int = 0
    total_processed: int = 0
    total_dropped: int = 0
    max_queue_size: int = 0
    avg_queue_size: float = 0.0


class TripleBuffer:
    """Triple buffer: recording, waiting, processing - xoay vòng liên tục.

    Đề tài yêu cầu:
    - Một buffer đang được ghi âm.
    - Một buffer đang chờ xử lý.
    - Một hoặc nhiều buffer đang được xử lý song song.

    Triển khai:
    - 3 buffer slots xoay vòng: [recording, waiting, processing]
    - Khi recording xong → chuyển sang waiting
    - Khi waiting đầy → chuyển sang processing queue
    - Worker pool xử lý song song các buffer từ processing queue
    """

    def __init__(self, chunk_samples: int, num_workers: int = 2):
        """
        Args:
            chunk_samples: Số samples cho mỗi chunk (VD: 16000 * 0.5 = 8000)
            num_workers: Số worker xử lý song song (đề tài gợi ý 1-3)
        """
        self.chunk_samples = chunk_samples
        self.num_workers = num_workers

        # 3 buffer slots xoay vòng
        self._buffers = [
            np.zeros(chunk_samples, dtype=np.float32),
            np.zeros(chunk_samples, dtype=np.float32),
            np.zeros(chunk_samples, dtype=np.float32),
        ]
        self._buffer_states = ['idle', 'idle', 'idle']  # idle, recording, waiting, processing, done

        # Con trỏ buffer hiện tại
        self._recording_idx = 0  # Buffer đang ghi
        self._waiting_idx = 1    # Buffer đang chờ
        self._processing_idx = 2  # Buffer đang/hay được xử lý

        # Vị trí ghi hiện tại trong buffer
        self._write_pos = 0

        # Queue chờ xử lý (có thể chứa nhiều chunks)
        self._process_queue: queue.Queue = queue.Queue(maxsize=10)

        # Lock để đồng bộ
        self._lock = threading.Lock()

        # Stats
        self._stats = BufferStats()
        self._queue_sizes: deque = deque(maxlen=1000)

        # Control
        self._running = False
        self._producer_thread: Optional[threading.Thread] = None
        self._worker_threads: List[threading.Thread] = []

        # Callback xử lý (MFCC, STT, etc.)
        self._process_callback: Optional[Callable] = None

        # ID counter
        self._chunk_id_counter = 0

    def set_process_callback(self, callback: Callable[[AudioChunk], Any]):
        """Set callback để xử lý mỗi chunk (VD: MFCC classify)."""
        self._process_callback = callback

    def _rotate_buffers(self):
        """Xoay buffer: recording → waiting → queue."""
        with self._lock:
            # Buffer đang ghi → chuyển sang waiting
            self._buffer_states[self._recording_idx] = 'waiting'

            # Buffer waiting đầy → đẩy vào queue
            if self._buffer_states[self._waiting_idx] == 'waiting':
                # Tạo chunk từ waiting buffer
                chunk = AudioChunk(
                    pcm=self._buffers[self._waiting_idx].copy(),
                    sample_rate=16000,
                    captured_at_ms=int(time.time() * 1000) - (self.chunk_samples / 16000 * 1000),
                    chunk_id=self._chunk_id_counter,
                )
                self._chunk_id_counter += 1

                # Reset waiting buffer
                self._buffers[self._waiting_idx].fill(0)
                self._buffer_states[self._waiting_idx] = 'idle'

                # Đẩy vào queue
                try:
                    self._process_queue.put_nowait(chunk)
                    self._stats.total_recorded += 1
                    self._queue_sizes.append(self._process_queue.qsize())
                    self._stats.max_queue_size = max(
                        self._stats.max_queue_size,
                        self._process_queue.qsize()
                    )
                except queue.Full:
                    self._stats.total_dropped += 1

            # Xoay index
            old_recording = self._recording_idx
            self._recording_idx = self._waiting_idx
            self._waiting_idx = self._processing_idx
            self._processing_idx = old_recording

            # Bắt đầu ghi buffer mới
            self._buffer_states[self._recording_idx] = 'recording'
            self._write_pos = 0

    def write_samples(self, samples: np.ndarray) -> bool:
        """Ghi samples vào buffer hiện tại.

        Args:
            samples: Float32 array, normalized [-1, 1]

        Returns:
            True nếu chunk hoàn tất và được đẩy vào queue
        """
        if not self._running:
            return False

        with self._lock:
            buf = self._buffers[self._recording_idx]

            for sample in samples:
                buf[self._write_pos] = sample
                self._write_pos += 1

                if self._write_pos >= self.chunk_samples:
                    # Chunk hoàn tất, xoay buffer
                    # Gọi _rotate_buffers bên ngoài lock để tránh deadlock
                    return True

            return False

    def _producer_loop(self, mic_callback: Callable[[int], Optional[np.ndarray]]):
        """Luồng producer: liên tục thu âm từ mic và ghi vào buffer.

        Args:
            mic_callback: Gọi với số samples cần, trả về numpy array hoặc None
        """
        samples_needed = 1024  # Thu 1024 samples mỗi lần

        while self._running:
            try:
                # Thu samples từ mic
                samples = mic_callback(samples_needed)
                if samples is not None and len(samples) > 0:
                    if self.write_samples(samples):
                        self._rotate_buffers()
                else:
                    time.sleep(0.001)  # Chờ 1ms nếu mic chưa có data
            except Exception as e:
                print(f'[TripleBuffer] Producer error: {e}')
                time.sleep(0.01)

    def _worker_loop(self, worker_id: int):
        """Luồng worker: lấy chunks từ queue và xử lý."""
        while self._running:
            try:
                # Lấy chunk từ queue
                chunk = self._process_queue.get(timeout=0.1)
                self._stats.total_processed += 1

                # Xử lý chunk
                if self._process_callback:
                    try:
                        self._process_callback(chunk)
                    except Exception as e:
                        print(f'[TripleBuffer] Worker {worker_id} process error: {e}')

                self._process_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f'[TripleBuffer] Worker {worker_id} error: {e}')

    def start(self, mic_callback: Callable[[int], Optional[np.ndarray]]):
        """Bắt đầu thu và xử lý."""
        if self._running:
            return

        self._running = True

        # Bắt đầu producer thread
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            args=(mic_callback,),
            daemon=True,
            name='TripleBuffer-Producer'
        )
        self._producer_thread.start()

        # Bắt đầu worker threads
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True,
                name=f'TripleBuffer-Worker-{i}'
            )
            t.start()
            self._worker_threads.append(t)

        print(f'[TripleBuffer] Started: {self.num_workers} workers, chunk={self.chunk_samples} samples')

    def stop(self):
        """Dừng tất cả threads."""
        self._running = False

        # Chờ threads kết thúc
        if self._producer_thread:
            self._producer_thread.join(timeout=1.0)
        for t in self._worker_threads:
            t.join(timeout=0.5)

        # Tính avg queue size
        if self._queue_sizes:
            self._stats.avg_queue_size = sum(self._queue_sizes) / len(self._queue_sizes)

        print(f'[TripleBuffer] Stopped: {self._stats}')

    def get_stats(self) -> BufferStats:
        """Lấy thống kê buffer."""
        if self._queue_sizes:
            self._stats.avg_queue_size = sum(self._queue_sizes) / len(self._queue_sizes)
        return self._stats

    @property
    def process_queue_size(self) -> int:
        """Số chunks đang chờ xử lý."""
        return self._process_queue.qsize()


class RingBuffer:
    """Ring buffer đơn giản cho audio streaming.

    Dùng khi không cần triple buffer phức tạp.
    """

    def __init__(self, max_size: int):
        self._buffer = np.zeros(max_size, dtype=np.float32)
        self._write_pos = 0
        self._read_pos = 0
        self._size = max_size
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> int:
        """Ghi data vào buffer.

        Returns:
            Số samples thực sự được ghi
        """
        with self._lock:
            available = self._size - self._write_pos
            to_write = min(len(data), available)
            self._buffer[self._write_pos:self._write_pos + to_write] = data[:to_write]
            self._write_pos = (self._write_pos + to_write) % self._size
            return to_write

    def read(self, n: int) -> np.ndarray:
        """Đọc n samples từ buffer."""
        with self._lock:
            n = min(n, self._size)
            if self._read_pos < self._write_pos:
                # Normal case
                data = self._buffer[self._read_pos:self._read_pos + n].copy()
            else:
                # Wrap around
                part1 = self._buffer[self._read_pos:]
                part2 = self._buffer[:n - len(part1)]
                data = np.concatenate([part1, part2])
            self._read_pos = (self._read_pos + n) % self._size
            return data

    def available(self) -> int:
        """Số samples có sẵn để đọc."""
        with self._lock:
            if self._write_pos >= self._read_pos:
                return self._write_pos - self._read_pos
            return self._size - self._read_pos + self._write_pos

    def clear(self):
        """Xóa buffer."""
        with self._lock:
            self._buffer.fill(0)
            self._write_pos = 0
            self._read_pos = 0
