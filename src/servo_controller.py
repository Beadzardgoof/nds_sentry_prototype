import time
import threading
from typing import Tuple, Optional
import numpy as np

class ServoController:
    """
    Controls two servos (pan/tilt) to track targets based on predicted coordinates.
    Handles coordinate transformation from normalized image coordinates to servo angles.
    """
    
    def __init__(self):
        # Servo configuration
        self.pan_servo_channel = 0  # Horizontal servo
        self.tilt_servo_channel = 1  # Vertical servo
        
        # Servo limits (in degrees)
        self.pan_min_angle = -90
        self.pan_max_angle = 90
        self.tilt_min_angle = -45
        self.tilt_max_angle = 45
        
        # Current servo positions
        self.current_pan_angle = 0.0
        self.current_tilt_angle = 0.0
        
        # Movement parameters
        self.max_speed_deg_per_sec = 180  # Maximum servo speed
        self.smoothing_factor = 0.7  # Smoothing for sudden movements
        self.deadband_threshold = 1.0  # Minimum movement threshold in degrees
        
        # Camera and servo calibration parameters
        self.camera_fov_horizontal = 90  # Horizontal field of view in degrees
        self.camera_fov_vertical = 60    # Vertical field of view in degrees
        
        # Servo communication (placeholder for actual servo driver)
        self.servo_driver = None
        self.control_lock = threading.Lock()
        
        # Initialize servo hardware (placeholder)
        self.initialize_servos()
        
    def initialize_servos(self) -> bool:
        """Initialize servo hardware communication"""
        try:
            # Placeholder for actual servo driver initialization
            # Examples: PWM board, Arduino communication, servo HAT, etc.
            # self.servo_driver = ServoDriver() 
            
            # Move to center position
            self.move_to_angle(0, 0)
            print("Servo controller initialized")
            return True
            
        except Exception as e:
            print(f"Error initializing servos: {e}")
            return False
            
    def move_to_target(self, target_coords: Tuple[float, float]):
        """
        Move servos to track target at given normalized coordinates
        Args:
            target_coords: (x, y) coordinates in normalized range [0, 1]
        """
        with self.control_lock:
            # Convert normalized coordinates to servo angles
            pan_angle, tilt_angle = self.coords_to_angles(target_coords)
            
            # Apply smoothing to prevent jerky movements
            smoothed_pan = self.apply_smoothing(pan_angle, self.current_pan_angle)
            smoothed_tilt = self.apply_smoothing(tilt_angle, self.current_tilt_angle)
            
            # Check if movement is significant enough
            if self.should_move(smoothed_pan, smoothed_tilt):
                self.move_to_angle(smoothed_pan, smoothed_tilt)
                
    def coords_to_angles(self, coords: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert normalized image coordinates to servo angles
        Args:
            coords: (x, y) in range [0, 1] where (0,0) is top-left, (1,1) is bottom-right
        Returns:
            (pan_angle, tilt_angle) in degrees
        """
        x, y = coords
        
        # Convert normalized coords to centered coords [-0.5, 0.5]
        centered_x = x - 0.5
        centered_y = 0.5 - y  # Flip Y axis (image Y=0 is top, servo Y=0 is center)
        
        # Convert to angles based on camera field of view
        pan_angle = centered_x * self.camera_fov_horizontal
        tilt_angle = centered_y * self.camera_fov_vertical
        
        # Clamp to servo limits
        pan_angle = np.clip(pan_angle, self.pan_min_angle, self.pan_max_angle)
        tilt_angle = np.clip(tilt_angle, self.tilt_min_angle, self.tilt_max_angle)
        
        return (pan_angle, tilt_angle)
        
    def apply_smoothing(self, target_angle: float, current_angle: float) -> float:
        """Apply smoothing filter to servo movements"""
        # Simple exponential smoothing
        smoothed = (self.smoothing_factor * target_angle + 
                   (1 - self.smoothing_factor) * current_angle)
        return smoothed
        
    def should_move(self, pan_angle: float, tilt_angle: float) -> bool:
        """Check if servo movement is significant enough to execute"""
        pan_diff = abs(pan_angle - self.current_pan_angle)
        tilt_diff = abs(tilt_angle - self.current_tilt_angle)
        
        return (pan_diff > self.deadband_threshold or 
                tilt_diff > self.deadband_threshold)
                
    def move_to_angle(self, pan_angle: float, tilt_angle: float):
        """
        Move servos to specific angles
        Args:
            pan_angle: Horizontal angle in degrees
            tilt_angle: Vertical angle in degrees
        """
        try:
            # Clamp angles to limits
            pan_angle = np.clip(pan_angle, self.pan_min_angle, self.pan_max_angle)
            tilt_angle = np.clip(tilt_angle, self.tilt_min_angle, self.tilt_max_angle)
            
            # Placeholder for actual servo movement
            # Examples of real implementations:
            # self.servo_driver.set_angle(self.pan_servo_channel, pan_angle)
            # self.servo_driver.set_angle(self.tilt_servo_channel, tilt_angle)
            
            # For debugging - print movement commands
            if abs(pan_angle - self.current_pan_angle) > 0.1 or abs(tilt_angle - self.current_tilt_angle) > 0.1:
                print(f"Servo move: Pan={pan_angle:.1f}°, Tilt={tilt_angle:.1f}°")
            
            # Update current positions
            self.current_pan_angle = pan_angle
            self.current_tilt_angle = tilt_angle
            
        except Exception as e:
            print(f"Error moving servos: {e}")
            
    def get_current_angles(self) -> Tuple[float, float]:
        """Get current servo angles"""
        return (self.current_pan_angle, self.current_tilt_angle)
        
    def center_servos(self):
        """Move servos to center position"""
        self.move_to_angle(0, 0)
        
    def set_servo_limits(self, pan_min: float, pan_max: float, tilt_min: float, tilt_max: float):
        """Set servo movement limits"""
        self.pan_min_angle = pan_min
        self.pan_max_angle = pan_max
        self.tilt_min_angle = tilt_min
        self.tilt_max_angle = tilt_max
        
    def set_camera_fov(self, horizontal_fov: float, vertical_fov: float):
        """Set camera field of view parameters for coordinate transformation"""
        self.camera_fov_horizontal = horizontal_fov
        self.camera_fov_vertical = vertical_fov
        
    def calibrate_servo_mapping(self, test_points: list):
        """
        Calibrate the mapping between image coordinates and servo angles
        Args:
            test_points: List of (image_coords, actual_angles) pairs for calibration
        """
        # Placeholder for calibration routine
        # This would involve moving servos to known positions and measuring
        # the resulting image coordinates to build an accurate mapping
        print("Servo calibration placeholder - implement based on hardware setup")
        
    def emergency_stop(self):
        """Emergency stop - center servos and disable movement"""
        print("Emergency stop activated")
        self.center_servos()
        
    def get_status(self) -> dict:
        """Get current servo controller status"""
        return {
            'pan_angle': self.current_pan_angle,
            'tilt_angle': self.current_tilt_angle,
            'pan_limits': (self.pan_min_angle, self.pan_max_angle),
            'tilt_limits': (self.tilt_min_angle, self.tilt_max_angle),
            'servo_driver_connected': self.servo_driver is not None
        }