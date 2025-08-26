import numpy as np
import time
from typing import Tuple, Optional, List, Deque
from collections import deque

class TargetPredictionModel:
    """
    High-accuracy target prediction model for when success rate is high.
    Uses recent target history to predict future positions with advanced algorithms.
    """
    
    def __init__(self, history_length: int = 10):
        self.history_length = history_length
        # Store recent target positions with timestamps
        self.target_history: Deque[Tuple[Tuple[float, float], float]] = deque(maxlen=history_length)
        self.velocity_history: Deque[Tuple[float, float]] = deque(maxlen=history_length-1)
        
        # Prediction parameters
        self.prediction_time_horizon = 0.1  # Predict 100ms ahead
        self.min_history_for_prediction = 3
        
    def predict(self, current_coords: Tuple[float, float], frame_label: float) -> Tuple[float, float]:
        """
        Predict target position using advanced prediction algorithms
        Args:
            current_coords: Current target coordinates (normalized 0-1)
            frame_label: Timestamp of the current frame
        Returns:
            Predicted target coordinates (normalized 0-1)
        """
        # Add current position to history
        self.target_history.append((current_coords, frame_label))
        
        # Calculate velocity if we have enough history
        if len(self.target_history) >= 2:
            self._update_velocity_history()
        
        # Return prediction based on available data
        if len(self.target_history) >= self.min_history_for_prediction:
            return self._advanced_prediction(current_coords, frame_label)
        else:
            # Not enough history - return current position with small smoothing
            return self._simple_prediction(current_coords)
            
    def _update_velocity_history(self):
        """Calculate and store velocity between recent positions"""
        if len(self.target_history) < 2:
            return
            
        # Get last two positions
        (x2, y2), t2 = self.target_history[-1]
        (x1, y1), t1 = self.target_history[-2]
        
        # Calculate velocity (change per second)
        dt = t2 - t1
        if dt > 0:
            vx = (x2 - x1) / dt
            vy = (y2 - y1) / dt
            self.velocity_history.append((vx, vy))
            
    def _simple_prediction(self, current_coords: Tuple[float, float]) -> Tuple[float, float]:
        """Simple prediction when insufficient history is available"""
        # Apply minimal smoothing filter
        if len(self.target_history) >= 2:
            (prev_x, prev_y), _ = self.target_history[-2]
            current_x, current_y = current_coords
            
            # Simple exponential smoothing
            alpha = 0.7
            smoothed_x = alpha * current_x + (1 - alpha) * prev_x
            smoothed_y = alpha * current_y + (1 - alpha) * prev_y
            
            return (smoothed_x, smoothed_y)
        
        return current_coords
        
    def _advanced_prediction(self, current_coords: Tuple[float, float], frame_label: float) -> Tuple[float, float]:
        """Advanced prediction using velocity and acceleration analysis"""
        current_x, current_y = current_coords
        
        # Method 1: Kalman-like prediction using velocity
        predicted_coords = self._velocity_prediction(current_coords, frame_label)
        
        # Method 2: Polynomial extrapolation for smooth trajectories
        if len(self.target_history) >= 5:
            poly_coords = self._polynomial_prediction(frame_label)
            # Blend predictions (favor velocity prediction for responsiveness)
            predicted_coords = self._blend_predictions(predicted_coords, poly_coords, weight_velocity=0.7)
        
        # Clamp coordinates to valid range [0, 1]
        predicted_x = np.clip(predicted_coords[0], 0.0, 1.0)
        predicted_y = np.clip(predicted_coords[1], 0.0, 1.0)
        
        return (predicted_x, predicted_y)
        
    def _velocity_prediction(self, current_coords: Tuple[float, float], frame_label: float) -> Tuple[float, float]:
        """Predict using current velocity with acceleration compensation"""
        if len(self.velocity_history) == 0:
            return current_coords
            
        current_x, current_y = current_coords
        
        # Get current velocity (average of recent velocities for stability)
        recent_velocities = list(self.velocity_history)[-3:]  # Use last 3 velocity measurements
        avg_vx = np.mean([v[0] for v in recent_velocities])
        avg_vy = np.mean([v[1] for v in recent_velocities])
        
        # Predict position based on velocity
        predicted_x = current_x + avg_vx * self.prediction_time_horizon
        predicted_y = current_y + avg_vy * self.prediction_time_horizon
        
        # Apply acceleration compensation if we have enough velocity history
        if len(self.velocity_history) >= 3:
            # Calculate acceleration
            (vx2, vy2) = self.velocity_history[-1]
            (vx1, vy1) = self.velocity_history[-2]
            
            # Simple acceleration estimate
            dt = 1.0 / 30.0  # Assume 30 FPS for dt estimation
            ax = (vx2 - vx1) / dt
            ay = (vy2 - vy1) / dt
            
            # Apply acceleration correction
            t = self.prediction_time_horizon
            predicted_x += 0.5 * ax * t * t
            predicted_y += 0.5 * ay * t * t
        
        return (predicted_x, predicted_y)
        
    def _polynomial_prediction(self, frame_label: float) -> Tuple[float, float]:
        """Predict using polynomial extrapolation for smooth trajectories"""
        if len(self.target_history) < 4:
            return self.target_history[-1][0]
            
        # Extract recent positions and times
        positions = [pos for pos, _ in self.target_history]
        times = [t for _, t in self.target_history]
        
        # Normalize times relative to current time
        current_time = times[-1]
        normalized_times = [(t - current_time) for t in times]
        
        # Fit polynomial to recent trajectory
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        try:
            # Fit quadratic polynomial (degree 2)
            poly_x = np.polyfit(normalized_times, x_coords, deg=2)
            poly_y = np.polyfit(normalized_times, y_coords, deg=2)
            
            # Predict at future time
            future_time = self.prediction_time_horizon
            predicted_x = np.polyval(poly_x, future_time)
            predicted_y = np.polyval(poly_y, future_time)
            
            return (predicted_x, predicted_y)
            
        except np.linalg.LinAlgError:
            # Fallback if polynomial fitting fails
            return self.target_history[-1][0]
            
    def _blend_predictions(self, pred1: Tuple[float, float], pred2: Tuple[float, float], 
                          weight_velocity: float = 0.7) -> Tuple[float, float]:
        """Blend two predictions with given weights"""
        w1 = weight_velocity
        w2 = 1.0 - weight_velocity
        
        blended_x = w1 * pred1[0] + w2 * pred2[0]
        blended_y = w1 * pred1[1] + w2 * pred2[1]
        
        return (blended_x, blended_y)
        
    def get_prediction_confidence(self) -> float:
        """Return confidence metric for current predictions"""
        if len(self.target_history) < self.min_history_for_prediction:
            return 0.3
            
        # Base confidence on velocity consistency
        if len(self.velocity_history) < 3:
            return 0.6
            
        # Calculate velocity variance as confidence metric
        recent_velocities = list(self.velocity_history)[-5:]
        vx_values = [v[0] for v in recent_velocities]
        vy_values = [v[1] for v in recent_velocities]
        
        vx_std = np.std(vx_values)
        vy_std = np.std(vy_values)
        
        # Higher variance means lower confidence
        velocity_variance = (vx_std + vy_std) / 2.0
        confidence = max(0.1, 1.0 - velocity_variance * 10.0)
        
        return min(1.0, confidence)