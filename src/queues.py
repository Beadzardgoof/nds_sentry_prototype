import queue
import threading
from typing import Optional
from .frame_data import FrameData, ProcessedFrameData

class FrameQueue:
    """Thread-safe queue for raw camera frames with frame labeling"""
    
    def __init__(self, maxsize: int = 10):
        self._queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        
    def put(self, frame_data: FrameData, timeout: Optional[float] = None):
        """Put a frame into the queue with optional timeout"""
        try:
            self._queue.put(frame_data, timeout=timeout)
        except queue.Full:
            # Drop oldest frame if queue is full (real-time processing)
            try:
                self._queue.get_nowait()
                self._queue.put(frame_data, timeout=timeout)
            except queue.Empty:
                pass
                
    def get(self, timeout: Optional[float] = None) -> FrameData:
        """Get a frame from the queue with optional timeout"""
        return self._queue.get(timeout=timeout)
        
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self._queue.empty()
        
    def qsize(self) -> int:
        """Get current queue size"""
        return self._queue.qsize()

class ProcessedFrameQueue:
    """Thread-safe queue for processed frames with target coordinates"""
    
    def __init__(self, maxsize: int = 10):
        self._queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        
    def put(self, processed_data: ProcessedFrameData, timeout: Optional[float] = None):
        """Put processed frame data into the queue with optional timeout"""
        try:
            self._queue.put(processed_data, timeout=timeout)
        except queue.Full:
            # Drop oldest processed frame if queue is full
            try:
                self._queue.get_nowait()
                self._queue.put(processed_data, timeout=timeout)
            except queue.Empty:
                pass
                
    def get(self, timeout: Optional[float] = None) -> ProcessedFrameData:
        """Get processed frame data from the queue with optional timeout"""
        return self._queue.get(timeout=timeout)
        
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self._queue.empty()
        
    def qsize(self) -> int:
        """Get current queue size"""
        return self._queue.qsize()