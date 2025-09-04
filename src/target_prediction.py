import numpy as np
import time
from typing import Tuple, Optional, List, Deque
from collections import deque

class KalmanTargetPredictor:
    """
    Kalman filter-based target prediction using angular position axioms.
    State vector: [θx, θy, θ̇x, θ̇y, θ̈x, θ̈y] - angular position, velocity, acceleration
    All angles are relative to camera reference frame at initialization.
    """
    
    def __init__(self, alpha: float = 0.1, process_noise_scale: float = 1e-4):
        # State vector: [θx, θy, θ̇x, θ̇y, θ̈x, θ̈y]
        self.state = np.zeros(6)  # Current state estimate
        self.P = np.eye(6) * 1000  # State covariance matrix (high initial uncertainty)
        
        # Kalman filter parameters
        self.alpha = alpha  # Measurement noise scaling factor
        self.process_noise_scale = process_noise_scale
        
        # Initialization tracking
        self.initialization_frames = []  # Store first 5 frames for initialization
        self.min_frames_for_init = 5
        self.is_initialized = False
        
        # Camera reference frame (set during initialization)
        self.camera_reference_position = None
        self.camera_current_position = np.zeros(2)  # Current camera angular position
        
        # Time tracking
        self.last_update_time = None
        
        # Field of view (degrees) - used for coordinate conversion
        self.fov_horizontal = 90.0
        self.fov_vertical = 90.0
        
    def normalized_to_angular(self, coords: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert normalized coordinates (0-1) to angular coordinates (-1 to 1).
        Maps to -180° to +180° range for servo control.
        """
        x_norm, y_norm = coords
        
        # Convert from [0,1] to [-1,1] range
        theta_x = (x_norm - 0.5) * 2.0
        theta_y = (y_norm - 0.5) * 2.0
        
        return (theta_x, theta_y)
    
    def angular_to_normalized(self, angles: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert angular coordinates (-1 to 1) back to normalized coordinates (0-1).
        """
        theta_x, theta_y = angles
        
        # Convert from [-1,1] to [0,1] range
        x_norm = theta_x / 2.0 + 0.5
        y_norm = theta_y / 2.0 + 0.5
        
        # Clamp to valid range
        x_norm = np.clip(x_norm, 0.0, 1.0)
        y_norm = np.clip(y_norm, 0.0, 1.0)
        
        return (x_norm, y_norm)
    
    def create_state_transition_matrix(self, dt: float) -> np.ndarray:
        """
        Create state transition matrix F for semi-constant acceleration model.
        State: [θx, θy, θ̇x, θ̇y, θ̈x, θ̈y]
        """
        F = np.eye(6)
        
        # Position updates from velocity and acceleration
        F[0, 2] = dt      # θx += θ̇x * dt
        F[1, 3] = dt      # θy += θ̇y * dt
        F[0, 4] = 0.5 * dt**2  # θx += 0.5 * θ̈x * dt²
        F[1, 5] = 0.5 * dt**2  # θy += 0.5 * θ̈y * dt²
        
        # Velocity updates from acceleration
        F[2, 4] = dt      # θ̇x += θ̈x * dt
        F[3, 5] = dt      # θ̇y += θ̈y * dt
        
        # Acceleration remains semi-constant (identity for acceleration states)
        
        return F
    
    def create_control_matrix(self) -> np.ndarray:
        """
        Create control matrix B for camera servo movement.
        Control vector contains camera's angular position, velocity, acceleration.
        """
        # Control affects position directly (camera movement relative to reference)
        B = np.zeros((6, 6))
        
        # Camera movement affects target's relative position
        B[0, 0] = -1.0  # Camera θx movement affects target θx oppositely
        B[1, 1] = -1.0  # Camera θy movement affects target θy oppositely
        B[2, 2] = -1.0  # Camera θ̇x affects target θ̇x
        B[3, 3] = -1.0  # Camera θ̇y affects target θ̇y
        B[4, 4] = -1.0  # Camera θ̈x affects target θ̈x
        B[5, 5] = -1.0  # Camera θ̈y affects target θ̈y
        
        return B
    
    def create_process_noise_matrix(self, dt: float) -> np.ndarray:
        """
        Create process noise covariance matrix Q.
        Assumes noise in acceleration with propagation to velocity and position.
        """
        Q = np.zeros((6, 6))
        
        # Process noise primarily in acceleration
        sigma_a = self.process_noise_scale
        
        # Position noise from acceleration uncertainty
        Q[0, 0] = (sigma_a * dt**4) / 4  # θx position noise
        Q[1, 1] = (sigma_a * dt**4) / 4  # θy position noise
        
        # Velocity noise from acceleration uncertainty
        Q[2, 2] = (sigma_a * dt**2) / 2  # θ̇x velocity noise
        Q[3, 3] = (sigma_a * dt**2) / 2  # θ̇y velocity noise
        
        # Acceleration noise (direct)
        Q[4, 4] = sigma_a  # θ̈x acceleration noise
        Q[5, 5] = sigma_a  # θ̈y acceleration noise
        
        # Cross-correlations
        Q[0, 2] = (sigma_a * dt**3) / 2  # θx-θ̇x correlation
        Q[2, 0] = Q[0, 2]
        Q[1, 3] = (sigma_a * dt**3) / 2  # θy-θ̇y correlation
        Q[3, 1] = Q[1, 3]
        Q[0, 4] = (sigma_a * dt**2) / 2  # θx-θ̈x correlation
        Q[4, 0] = Q[0, 4]
        Q[1, 5] = (sigma_a * dt**2) / 2  # θy-θ̈y correlation
        Q[5, 1] = Q[1, 5]
        Q[2, 4] = sigma_a * dt  # θ̇x-θ̈x correlation
        Q[4, 2] = Q[2, 4]
        Q[3, 5] = sigma_a * dt  # θ̇y-θ̈y correlation
        Q[5, 3] = Q[3, 5]
        
        return Q
    
    def create_measurement_noise_matrix(self, yolo_certainty: float) -> np.ndarray:
        """
        Create measurement noise covariance matrix R based on YOLO certainty.
        R = alpha / (yolo_certainty^2)
        """
        # Prevent division by zero
        certainty_squared = max(yolo_certainty**2, 1e-6)
        noise_variance = self.alpha / certainty_squared
        
        R = np.eye(2) * noise_variance
        return R
    
    def initialize_from_frames(self, frames_data: List[Tuple[Tuple[float, float], float, float]]):
        """
        Initialize Kalman filter state from first 5 frames.
        frames_data: List of (coords, timestamp, yolo_certainty) tuples
        """
        if len(frames_data) < self.min_frames_for_init:
            return False
        
        # Convert coordinates to angular and extract data
        angular_positions = []
        timestamps = []
        
        for coords, timestamp, _ in frames_data:
            angular_pos = self.normalized_to_angular(coords)
            angular_positions.append(angular_pos)
            timestamps.append(timestamp)
        
        # Set camera reference frame from first detection
        if self.camera_reference_position is None:
            self.camera_reference_position = np.array([0.0, 0.0])  # Reference frame origin
        
        # Estimate initial position, velocity, and acceleration
        positions = np.array(angular_positions)
        times = np.array(timestamps)
        
        # Initial position (average of first few frames for stability)
        init_pos = np.mean(positions[:3], axis=0)
        
        # Estimate initial velocity using finite differences
        if len(positions) >= 3:
            dt1 = times[1] - times[0]
            dt2 = times[2] - times[1]
            
            if dt1 > 0 and dt2 > 0:
                vel1 = (positions[1] - positions[0]) / dt1
                vel2 = (positions[2] - positions[1]) / dt2
                init_vel = (vel1 + vel2) / 2.0
            else:
                init_vel = np.zeros(2)
        else:
            init_vel = np.zeros(2)
        
        # Estimate initial acceleration
        if len(positions) >= 4:
            dt3 = times[3] - times[2]
            if dt3 > 0:
                vel3 = (positions[3] - positions[2]) / dt3
                init_acc = (vel3 - vel2) / dt3 if dt2 > 0 else np.zeros(2)
            else:
                init_acc = np.zeros(2)
        else:
            init_acc = np.zeros(2)
        
        # Initialize state vector: [θx, θy, θ̇x, θ̇y, θ̈x, θ̈y]
        self.state[0:2] = init_pos
        self.state[2:4] = init_vel
        self.state[4:6] = init_acc
        
        # Reduce initial uncertainty now that we have data
        self.P = np.eye(6) * 10.0
        
        # Set initialization time
        self.last_update_time = timestamps[-1]
        self.is_initialized = True
        
        return True
    
    def predict(self, current_coords: Tuple[float, float], frame_label: float, 
                yolo_certainty: float = 0.8, camera_control: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Predict target position using Kalman filter.
        
        Args:
            current_coords: Current target coordinates (normalized 0-1)
            frame_label: Timestamp of the current frame
            yolo_certainty: YOLO detection certainty (0-1)
            camera_control: Camera angular position/velocity/acceleration [θx, θy, θ̇x, θ̇y, θ̈x, θ̈y]
        
        Returns:
            Predicted target coordinates (normalized 0-1)
        """
        # Store frame for initialization if not yet initialized
        if not self.is_initialized:
            self.initialization_frames.append((current_coords, frame_label, yolo_certainty))
            
            # Try to initialize if we have enough frames
            if len(self.initialization_frames) >= self.min_frames_for_init:
                if self.initialize_from_frames(self.initialization_frames):
                    # Clear initialization frames after successful initialization
                    self.initialization_frames = []
                else:
                    # Return current coordinates if initialization fails
                    return current_coords
            else:
                # Not enough frames for initialization yet
                return current_coords
        
        # Calculate time step
        dt = frame_label - self.last_update_time if self.last_update_time is not None else 1.0/30.0
        dt = max(dt, 1e-6)  # Prevent zero or negative dt
        
        # Prediction step
        F = self.create_state_transition_matrix(dt)
        Q = self.create_process_noise_matrix(dt)
        
        # Apply control if provided (camera movement)
        if camera_control is not None:
            B = self.create_control_matrix()
            self.state = F @ self.state + B @ camera_control
        else:
            self.state = F @ self.state
        
        self.P = F @ self.P @ F.T + Q
        
        # Update step
        # Convert current observation to angular coordinates
        angular_obs = np.array(self.normalized_to_angular(current_coords))
        
        # Measurement matrix (observe only position)
        H = np.zeros((2, 6))
        H[0, 0] = 1.0  # Observe θx
        H[1, 1] = 1.0  # Observe θy
        
        # Measurement noise
        R = self.create_measurement_noise_matrix(yolo_certainty)
        
        # Innovation
        y = angular_obs - H @ self.state
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            K = np.zeros((6, 2))
        
        # State update
        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        
        # Update timestamp
        self.last_update_time = frame_label
        
        # Return prediction (current state position converted back to normalized)
        predicted_angular = (self.state[0], self.state[1])
        return self.angular_to_normalized(predicted_angular)
    
    def get_prediction_confidence(self) -> float:
        """Return confidence metric for current predictions"""
        if not self.is_initialized:
            return 0.0
        
        # Base confidence on position uncertainty
        pos_uncertainty = np.sqrt(self.P[0, 0] + self.P[1, 1])
        
        # Convert to confidence (higher uncertainty = lower confidence)
        confidence = max(0.1, 1.0 - pos_uncertainty)
        return min(1.0, confidence)
    
    def update_camera_position(self, camera_angular_position: Tuple[float, float]):
        """Update current camera position relative to reference frame"""
        self.camera_current_position = np.array(camera_angular_position)


# Legacy alias for compatibility
TargetPredictionModel = KalmanTargetPredictor