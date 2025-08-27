import numpy as np
import time
from typing import Tuple, Optional, List, Deque
from collections import deque

class TargetExtrapolationModel:
    """
    Fallback target extrapolation model for low success rate scenarios.
    Uses aggressive prediction and pattern recognition to maintain tracking
    when target detection is unreliable.
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
        
    def extrapolate(self, current_coords: Optional[Tuple[float, float]], frame_label: float) -> Tuple[float, float]:
        """
        Extrapolate target position using aggressive prediction algorithms
        Args:
            current_coords: Current target coordinates (None if detection failed)
            frame_label: Timestamp of the current frame
        Returns:
            Extrapolated target coordinates (normalized 0-1)
        """
        current_time = time.time()
        
        if current_coords is not None:
            # We have a detection - update history and reset counters
            self.target_history.append((current_coords, frame_label))
            self.last_detection_time = current_time
            self.missed_detections = 0
            self._update_velocity_history()
            return self._compute_extrapolation(current_coords, frame_label)
        else:
            # No detection - use extrapolation based on previous data
            self.missed_detections += 1
            time_since_detection = current_time - self.last_detection_time
            
            return self._extrapolate_from_history(frame_label, time_since_detection)
            
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
            
    def _compute_extrapolation(self, current_coords: Tuple[float, float], frame_label: float) -> Tuple[float, float]:
        """Compute extrapolation when we have current detection"""
        if len(self.target_history) < self.min_history_for_extrapolation:
            return current_coords
            
        # Use more aggressive prediction than the high-accuracy model
        extrapolated = self._aggressive_velocity_extrapolation(current_coords)
        
        # Apply pattern-based correction if we have enough history
        if len(self.target_history) >= 5:
            pattern_correction = self._pattern_based_extrapolation()
            # Blend with pattern correction
            extrapolated = self._blend_extrapolations(extrapolated, pattern_correction, 0.6)
            
        # Apply boundary constraints with bounce prediction
        extrapolated = self._apply_boundary_constraints(extrapolated)
        
        return extrapolated
        
    def _extrapolate_from_history(self, frame_label: float, time_since_detection: float) -> Tuple[float, float]:
        """Extrapolate when no current detection is available"""
        if len(self.target_history) == 0:
            # No history available - return center of frame as fallback
            return (0.5, 0.5)
            
        # Get last known position
        last_pos, last_time = self.target_history[-1]
        
        # Use multiple extrapolation methods and blend them
        methods = []
        
        # Method 1: Velocity-based extrapolation
        if len(self.velocity_history) > 0:
            velocity_extrap = self._velocity_based_extrapolation(last_pos, time_since_detection)
            methods.append((velocity_extrap, 0.4))
            
        # Method 2: Pattern-based extrapolation
        if len(self.pattern_memory) > 3:
            pattern_extrap = self._pattern_memory_extrapolation(last_pos, time_since_detection)
            methods.append((pattern_extrap, 0.3))
            
        # Method 3: Spiral search pattern (for completely lost targets)
        if self.missed_detections > 10:
            spiral_pos = self._spiral_search_pattern(last_pos, time_since_detection)
            methods.append((spiral_pos, 0.3))
        
        # Blend all available methods
        if methods:
            return self._blend_multiple_extrapolations(methods)
        else:
            # Fallback to last known position with small random offset
            return self._last_position_fallback(last_pos)
            
    def _aggressive_velocity_extrapolation(self, current_coords: Tuple[float, float]) -> Tuple[float, float]:
        """More aggressive velocity-based extrapolation"""
        if len(self.velocity_history) == 0:
            return current_coords
            
        current_x, current_y = current_coords
        
        # Use longer velocity average for stability in noisy conditions
        recent_velocities = list(self.velocity_history)[-5:]
        
        # Weight recent velocities more heavily
        weights = np.exp(np.linspace(-1, 0, len(recent_velocities)))
        weights /= weights.sum()
        
        weighted_vx = sum(v[0] * w for v, w in zip(recent_velocities, weights))
        weighted_vy = sum(v[1] * w for v, w in zip(recent_velocities, weights))
        
        # Apply longer time horizon for aggressive prediction
        t = self.extrapolation_time_horizon
        
        # Add acceleration if available
        if len(self.velocity_history) >= 3:
            (vx2, vy2) = self.velocity_history[-1]
            (vx1, vy1) = self.velocity_history[-2]
            
            dt = 1.0 / 30.0  # Assume 30 FPS
            ax = (vx2 - vx1) / dt
            ay = (vy2 - vy1) / dt
            
            # Apply acceleration with longer time horizon
            predicted_x = current_x + weighted_vx * t + 0.5 * ax * t * t
            predicted_y = current_y + weighted_vy * t + 0.5 * ay * t * t
        else:
            predicted_x = current_x + weighted_vx * t
            predicted_y = current_y + weighted_vy * t
            
        return (predicted_x, predicted_y)
        
    def _pattern_based_extrapolation(self) -> Tuple[float, float]:
        """Extrapolate based on recognized movement patterns"""
        if len(self.target_history) < 5:
            return self.target_history[-1][0]
            
        # Analyze recent movement for patterns
        positions = [pos for pos, _ in list(self.target_history)[-10:]]
        
        # Check for circular/orbital patterns
        circular_center = self._detect_circular_pattern(positions)
        if circular_center is not None:
            return self._extrapolate_circular_motion(circular_center, positions)
            
        # Check for oscillating patterns
        oscillation_params = self._detect_oscillation_pattern(positions)
        if oscillation_params is not None:
            return self._extrapolate_oscillation(*oscillation_params)
            
        # Fallback to linear extrapolation
        return self._linear_pattern_extrapolation(positions)
        
    def _velocity_based_extrapolation(self, last_pos: Tuple[float, float], time_elapsed: float) -> Tuple[float, float]:
        """Extrapolate based on last known velocity"""
        if len(self.velocity_history) == 0:
            return last_pos
            
        # Use average of recent velocities
        recent_velocities = list(self.velocity_history)[-3:]
        avg_vx = np.mean([v[0] for v in recent_velocities])
        avg_vy = np.mean([v[1] for v in recent_velocities])
        
        # Extrapolate position
        extrap_x = last_pos[0] + avg_vx * time_elapsed
        extrap_y = last_pos[1] + avg_vy * time_elapsed
        
        return (extrap_x, extrap_y)
        
    def _spiral_search_pattern(self, center_pos: Tuple[float, float], time_elapsed: float) -> Tuple[float, float]:
        """Generate spiral search pattern for completely lost targets"""
        # Create expanding spiral search pattern
        spiral_radius = min(0.1 + time_elapsed * 0.05, 0.3)  # Expand search radius over time
        spiral_angle = (time_elapsed * 2.0) % (2 * np.pi)  # Rotate over time
        
        spiral_x = center_pos[0] + spiral_radius * np.cos(spiral_angle)
        spiral_y = center_pos[1] + spiral_radius * np.sin(spiral_angle)
        
        return (spiral_x, spiral_y)
        
    def _apply_boundary_constraints(self, coords: Tuple[float, float]) -> Tuple[float, float]:
        """Apply boundary constraints with bounce prediction"""
        x, y = coords
        
        # Simple bounce at boundaries
        if x < 0:
            x = -x  # Bounce off left wall
        elif x > 1:
            x = 2 - x  # Bounce off right wall
            
        if y < 0:
            y = -y  # Bounce off top wall
        elif y > 1:
            y = 2 - y  # Bounce off bottom wall
            
        # Clamp to valid range as final safety
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        return (x, y)
        
    def _blend_extrapolations(self, extrap1: Tuple[float, float], extrap2: Tuple[float, float], 
                             weight1: float) -> Tuple[float, float]:
        """Blend two extrapolations"""
        weight2 = 1.0 - weight1
        
        blended_x = weight1 * extrap1[0] + weight2 * extrap2[0]
        blended_y = weight1 * extrap1[1] + weight2 * extrap2[1]
        
        return (blended_x, blended_y)
        
    def _blend_multiple_extrapolations(self, methods: List[Tuple[Tuple[float, float], float]]) -> Tuple[float, float]:
        """Blend multiple extrapolation methods"""
        total_weight = sum(weight for _, weight in methods)
        
        if total_weight == 0:
            return (0.5, 0.5)
            
        weighted_x = sum(pos[0] * weight for pos, weight in methods) / total_weight
        weighted_y = sum(pos[1] * weight for pos, weight in methods) / total_weight
        
        return (weighted_x, weighted_y)
        
    # Placeholder methods for pattern recognition (to be implemented based on specific needs)
    def _detect_circular_pattern(self, positions: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """Detect circular movement patterns - placeholder"""
        return None
        
    def _detect_oscillation_pattern(self, positions: List[Tuple[float, float]]) -> Optional[Tuple]:
        """Detect oscillating movement patterns - placeholder"""
        return None
        
    def _extrapolate_circular_motion(self, center: Tuple[float, float], positions: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Extrapolate circular motion - placeholder"""
        return positions[-1]
        
    def _extrapolate_oscillation(self, *params) -> Tuple[float, float]:
        """Extrapolate oscillating motion - placeholder"""
        return self.target_history[-1][0] if self.target_history else (0.5, 0.5)
        
    def _linear_pattern_extrapolation(self, positions: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Simple linear extrapolation"""
        if len(positions) < 2:
            return positions[-1]
            
        # Use last two positions for linear extrapolation
        (x2, y2) = positions[-1]
        (x1, y1) = positions[-2]
        
        # Extrapolate forward
        t = self.extrapolation_time_horizon / 0.033  # Assume ~30 FPS
        extrap_x = x2 + (x2 - x1) * t
        extrap_y = y2 + (y2 - y1) * t
        
        return (extrap_x, extrap_y)
        
    def _pattern_memory_extrapolation(self, last_pos: Tuple[float, float], time_elapsed: float) -> Tuple[float, float]:
        """Extrapolate based on stored patterns - placeholder"""
        return last_pos
        
    def _last_position_fallback(self, last_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Fallback to last position with small random offset to aid reacquisition"""
        x, y = last_pos
        
        # Add small random offset to help reacquire target
        noise_level = 0.02 * (1 + self.missed_detections * 0.1)  # Increase noise over time
        
        x += np.random.uniform(-noise_level, noise_level)
        y += np.random.uniform(-noise_level, noise_level)
        
        return (np.clip(x, 0.0, 1.0), np.clip(y, 0.0, 1.0))
        
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