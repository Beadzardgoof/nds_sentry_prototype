import numpy as np
import time
import math
from typing import Tuple, Optional, List, Deque
from collections import deque
from .frame_data import ScanPattern, ScanPatternType

class TargetExtrapolationModel:
    """
    Fallback target extrapolation model for low success rate scenarios.
    Generates scan patterns for target search when detection fails.
    """
    
    def __init__(self, history_length: int = 20):
        self.history_length = history_length
        # Store longer history for pattern recognition
        self.target_history: Deque[Tuple[Tuple[float, float], float]] = deque(maxlen=history_length)
        self.velocity_history: Deque[Tuple[float, float]] = deque(maxlen=history_length-1)
        
        # Extrapolation parameters (more aggressive than prediction model)
        self.extrapolation_time_horizon = 0.2  # Predict 200ms ahead
        self.min_history_for_extrapolation = 2
        self.confidence_decay_rate = 0.95  # How quickly confidence decreases without new detections
        
        # Pattern recognition
        self.last_detection_time = time.time()
        self.missed_detections = 0
        self.pattern_memory = deque(maxlen=50)  # Store longer patterns
        
    def generate_scan_pattern(self, current_coords: Optional[Tuple[float, float]], frame_label: float) -> ScanPattern:
        """
        Generate scan pattern for target search when detection fails
        Args:
            current_coords: Current target coordinates (None if detection failed)
            frame_label: Timestamp of the current frame
        Returns:
            ScanPattern object with search parameters
        """
        current_time = time.time()
        
        if current_coords is not None:
            # We have a detection - update history and reset counters
            self.target_history.append((current_coords, frame_label))
            self.last_detection_time = current_time
            self.missed_detections = 0
            self._update_velocity_history()
            
            # Generate predicted path pattern based on current trajectory
            return self._generate_predicted_path_pattern(current_coords, frame_label)
        else:
            # No detection - generate search pattern based on previous data
            self.missed_detections += 1
            time_since_detection = current_time - self.last_detection_time
            
            return self._generate_search_pattern(frame_label, time_since_detection)
            
    def _update_velocity_history(self):
        """Calculate and store velocity between recent positions"""
        if len(self.target_history) < 2:
            return
            
        # Get last two positions
        (x2, y2), t2 = self.target_history[-1]
        (x1, y1), t1 = self.target_history[-2]
        
        # Calculate velocity
        dt = t2 - t1
        if dt > 0:
            vx = (x2 - x1) / dt
            vy = (y2 - y1) / dt
            self.velocity_history.append((vx, vy))

    def _generate_predicted_path_pattern(self, current_coords: Tuple[float, float], frame_label: float) -> ScanPattern:
        """Generate predicted path pattern when we have current detection"""
        if len(self.velocity_history) == 0:
            # No velocity data - use circular search around current position
            return ScanPattern(
                pattern_type=ScanPatternType.CIRCULAR,
                center_coords=current_coords,
                search_radius=0.1,
                scan_speed=2.0,
                direction=0.0,
                confidence=0.6,
                timestamp=time.time()
            )
        
        # Calculate predicted trajectory direction
        recent_vx = np.mean([v[0] for v in list(self.velocity_history)[-3:]])
        recent_vy = np.mean([v[1] for v in list(self.velocity_history)[-3:]])
        
        trajectory_direction = math.atan2(recent_vy, recent_vx)
        velocity_magnitude = math.sqrt(recent_vx**2 + recent_vy**2)
        
        # Create predicted path pattern
        return ScanPattern(
            pattern_type=ScanPatternType.PREDICTED_PATH,
            center_coords=current_coords,
            search_radius=min(0.2, velocity_magnitude * 2.0),  # Scale with velocity
            scan_speed=1.5,
            direction=trajectory_direction,
            confidence=0.8,
            timestamp=time.time()
        )

    def _generate_search_pattern(self, frame_label: float, time_since_detection: float) -> ScanPattern:
        """Generate search pattern when no detection is available"""
        if len(self.target_history) == 0:
            # No history - start with center search
            return ScanPattern(
                pattern_type=ScanPatternType.SPIRAL,
                center_coords=(0.5, 0.5),
                search_radius=0.3,
                scan_speed=1.0,
                direction=0.0,
                confidence=0.3,
                timestamp=time.time()
            )
        
        # Get last known position
        last_pos, _ = self.target_history[-1]
        
        # Choose pattern based on time since detection and missed detections
        if time_since_detection < 1.0:
            # Recently lost - use circular search around last position
            return self._generate_circular_search(last_pos, time_since_detection)
        elif self.missed_detections < 20:
            # Moderately lost - use spiral search expanding from last position
            return self._generate_spiral_search(last_pos, time_since_detection)
        else:
            # Long lost - use grid search pattern
            return self._generate_grid_search(last_pos, time_since_detection)

    def _generate_circular_search(self, last_pos: Tuple[float, float], time_elapsed: float) -> ScanPattern:
        """Generate circular search pattern around last known position"""
        # Expand search radius with time
        search_radius = min(0.1 + time_elapsed * 0.05, 0.25)
        
        return ScanPattern(
            pattern_type=ScanPatternType.CIRCULAR,
            center_coords=last_pos,
            search_radius=search_radius,
            scan_speed=3.0,  # Fast circular scan
            direction=0.0,
            confidence=0.7 - (time_elapsed * 0.1),  # Decrease confidence with time
            timestamp=time.time()
        )

    def _generate_spiral_search(self, last_pos: Tuple[float, float], time_elapsed: float) -> ScanPattern:
        """Generate spiral search pattern expanding from last known position"""
        # Larger search area for spiral
        search_radius = min(0.15 + time_elapsed * 0.03, 0.4)
        
        return ScanPattern(
            pattern_type=ScanPatternType.SPIRAL,
            center_coords=last_pos,
            search_radius=search_radius,
            scan_speed=2.0,  # Moderate spiral speed
            direction=0.0,
            confidence=0.5 - (time_elapsed * 0.05),
            timestamp=time.time()
        )

    def _generate_grid_search(self, last_pos: Tuple[float, float], time_elapsed: float) -> ScanPattern:
        """Generate systematic grid search pattern"""
        # Large search area for grid
        search_radius = min(0.25 + time_elapsed * 0.02, 0.5)
        
        return ScanPattern(
            pattern_type=ScanPatternType.GRID,
            center_coords=last_pos,
            search_radius=search_radius,
            scan_speed=1.5,  # Systematic grid speed
            direction=0.0,
            confidence=0.4 - (time_elapsed * 0.02),
            timestamp=time.time()
        )

    def get_extrapolation_confidence(self) -> float:
        """Return confidence metric for extrapolations"""
        # Confidence decreases with time since last detection
        time_since_detection = time.time() - self.last_detection_time
        time_decay = self.confidence_decay_rate ** time_since_detection
        
        # Confidence decreases with missed detections
        miss_penalty = 0.9 ** self.missed_detections
        
        # Base confidence on available history
        history_confidence = min(1.0, len(self.target_history) / 10.0)
        
        return time_decay * miss_penalty * history_confidence
    
    # Legacy method for backwards compatibility
    def extrapolate(self, current_coords: Optional[Tuple[float, float]], frame_label: float) -> Tuple[float, float]:
        """Legacy method - converts scan pattern to center coordinates for backwards compatibility"""
        pattern = self.generate_scan_pattern(current_coords, frame_label)
        return pattern.center_coords
            
