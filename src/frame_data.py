from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import time
from enum import Enum

class ScanPatternType(Enum):
    CIRCULAR = "circular"
    SPIRAL = "spiral" 
    GRID = "grid"
    LINEAR_SWEEP = "linear_sweep"
    PREDICTED_PATH = "predicted_path"

@dataclass
class ScanPattern:
    """Scan pattern data for extrapolation mode"""
    pattern_type: ScanPatternType
    center_coords: Tuple[float, float]  # (x, y) center point
    search_radius: float  # Search area radius
    scan_speed: float  # Pattern execution speed
    direction: float  # Pattern direction in radians
    confidence: float  # Pattern likelihood
    timestamp: float
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class FrameData:
    """Data structure for raw camera frames with labeling"""
    frame: np.ndarray
    frame_label: float  # Timestamp for frame synchronization
    timestamp: float
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def is_fresh(self, max_age: float = 0.1) -> bool:
        """Check if frame is fresh enough for processing"""
        current_time = time.time()
        return (current_time - self.timestamp) <= max_age

@dataclass 
class ProcessedFrameData:
    """Data structure for processed frames with target coordinates or scan patterns"""
    target_coords: Optional[Tuple[float, float]]  # (x, y) coordinates - None if scan pattern
    processed_frame_label: float  # Corresponding to original frame_label
    confidence: float  # Detection confidence score or pattern confidence
    timestamp: float
    scan_pattern: Optional[ScanPattern] = None  # Scan pattern data - None if coordinates
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def is_fresh(self, max_age: float = 0.1) -> bool:
        """Check if processed data is fresh enough for servo control"""
        current_time = time.time()
        return (current_time - self.timestamp) <= max_age
    
    def has_coordinates(self) -> bool:
        """Check if this data contains target coordinates"""
        return self.target_coords is not None
    
    def has_scan_pattern(self) -> bool:
        """Check if this data contains scan pattern"""
        return self.scan_pattern is not None
    
    def is_frame_fresh(self, frame_label_tolerance: float = 0.05) -> bool:
        """Check if the frame label indicates fresh processing"""
        current_frame_time = time.time()
        return abs(current_frame_time - self.processed_frame_label) <= frame_label_tolerance