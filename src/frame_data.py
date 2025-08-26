from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import time

@dataclass
class FrameData:
    """Data structure for raw camera frames with labeling"""
    frame: np.ndarray
    frame_label: float  # Timestamp for frame synchronization
    timestamp: float
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass 
class ProcessedFrameData:
    """Data structure for processed frames with target coordinates"""
    target_coords: Optional[Tuple[float, float]]  # (x, y) coordinates
    processed_frame_label: float  # Corresponding to original frame_label
    confidence: float  # Detection confidence score
    timestamp: float
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()