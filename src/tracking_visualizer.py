import cv2
import numpy as np
import time
from typing import Tuple, Optional, List, Deque
from collections import deque

class TrackingVisualizer:
    """
    Real-time visualization of target tracking with overlay graphics.
    Shows target position, tracking history, prediction trails, and camera aim point.
    """
    
    def __init__(self, max_history: int = 50, trail_length: int = 20):
        self.max_history = max_history
        self.trail_length = trail_length
        
        # Tracking history storage
        self.target_history: Deque[Tuple[float, float, float]] = deque(maxlen=max_history)  # (x, y, timestamp)
        self.prediction_history: Deque[Tuple[float, float, float]] = deque(maxlen=trail_length)  # (x, y, timestamp)
        self.camera_position = (0.5, 0.5)  # Current camera aim point (normalized)
        
        # Visualization settings
        self.colors = {
            'target_detected': (0, 255, 0),      # Green for detected target
            'target_predicted': (0, 255, 255),   # Yellow for predicted position
            'camera_aim': (0, 0, 255),           # Red for camera aim point
            'trail_detected': (100, 255, 100),   # Light green for detection trail
            'trail_predicted': (100, 255, 255),  # Light yellow for prediction trail
            'text': (255, 255, 255),             # White for text
            'crosshair': (255, 0, 255),          # Magenta for crosshairs
        }
        
        # Display parameters
        self.dot_size = 8
        self.trail_dot_size = 4
        self.camera_dot_size = 10
        self.line_thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
        # Frame size (will be set when first frame is processed)
        self.frame_width = None
        self.frame_height = None
        
    def normalize_to_pixel(self, norm_coords: Tuple[float, float]) -> Tuple[int, int]:
        """Convert normalized coordinates (0-1) to pixel coordinates"""
        if self.frame_width is None or self.frame_height is None:
            return (0, 0)
        
        x_norm, y_norm = norm_coords
        x_pixel = int(x_norm * self.frame_width)
        y_pixel = int(y_norm * self.frame_height)
        
        # Clamp to frame boundaries
        x_pixel = max(0, min(x_pixel, self.frame_width - 1))
        y_pixel = max(0, min(y_pixel, self.frame_height - 1))
        
        return (x_pixel, y_pixel)
    
    def update_target_position(self, coords: Tuple[float, float], is_detected: bool, timestamp: float = None):
        """Update target position in tracking history"""
        if timestamp is None:
            timestamp = time.time()
        
        if is_detected:
            # Store detected position
            self.target_history.append((coords[0], coords[1], timestamp))
        else:
            # Store predicted position
            self.prediction_history.append((coords[0], coords[1], timestamp))
    
    def update_camera_position(self, coords: Tuple[float, float]):
        """Update camera aim position"""
        self.camera_position = coords
    
    def draw_crosshairs(self, frame: np.ndarray):
        """Draw crosshairs at frame center"""
        if self.frame_height is None or self.frame_width is None:
            return
        
        center_x = self.frame_width // 2
        center_y = self.frame_height // 2
        crosshair_size = 20
        
        # Horizontal line
        cv2.line(frame, 
                (center_x - crosshair_size, center_y), 
                (center_x + crosshair_size, center_y), 
                self.colors['crosshair'], self.line_thickness)
        
        # Vertical line
        cv2.line(frame, 
                (center_x, center_y - crosshair_size), 
                (center_x, center_y + crosshair_size), 
                self.colors['crosshair'], self.line_thickness)
    
    def draw_tracking_trail(self, frame: np.ndarray):
        """Draw tracking history trail"""
        current_time = time.time()
        trail_max_age = 2.0  # Show 2 seconds of history
        
        # Draw detected target trail
        if len(self.target_history) > 1:
            trail_points = []
            for i, (x, y, timestamp) in enumerate(self.target_history):
                age = current_time - timestamp
                if age <= trail_max_age:
                    pixel_pos = self.normalize_to_pixel((x, y))
                    trail_points.append(pixel_pos)
            
            # Draw trail lines
            if len(trail_points) > 1:
                for i in range(1, len(trail_points)):
                    # Fade older points
                    alpha = 1.0 - (i / len(trail_points)) * 0.7
                    color = tuple(int(c * alpha) for c in self.colors['trail_detected'])
                    cv2.line(frame, trail_points[i-1], trail_points[i], color, 1)
        
        # Draw prediction trail
        if len(self.prediction_history) > 1:
            pred_points = []
            for x, y, timestamp in self.prediction_history:
                age = current_time - timestamp
                if age <= trail_max_age:
                    pixel_pos = self.normalize_to_pixel((x, y))
                    pred_points.append(pixel_pos)
            
            # Draw prediction trail (dashed effect)
            if len(pred_points) > 1:
                for i in range(1, len(pred_points)):
                    if i % 2 == 0:  # Dashed line effect
                        alpha = 0.8
                        color = tuple(int(c * alpha) for c in self.colors['trail_predicted'])
                        cv2.line(frame, pred_points[i-1], pred_points[i], color, 1)
    
    def draw_target_indicators(self, frame: np.ndarray, 
                             current_target: Optional[Tuple[float, float]] = None,
                             is_detected: bool = False):
        """Draw current target position indicator"""
        if current_target is not None:
            pixel_pos = self.normalize_to_pixel(current_target)
            
            if is_detected:
                # Green circle for detected target
                cv2.circle(frame, pixel_pos, self.dot_size, self.colors['target_detected'], -1)
                cv2.circle(frame, pixel_pos, self.dot_size + 2, self.colors['target_detected'], 2)
                
                # Add small cross inside
                cv2.line(frame, 
                        (pixel_pos[0] - 4, pixel_pos[1]), 
                        (pixel_pos[0] + 4, pixel_pos[1]), 
                        (0, 0, 0), 1)
                cv2.line(frame, 
                        (pixel_pos[0], pixel_pos[1] - 4), 
                        (pixel_pos[0], pixel_pos[1] + 4), 
                        (0, 0, 0), 1)
            else:
                # Yellow circle for predicted target
                cv2.circle(frame, pixel_pos, self.dot_size, self.colors['target_predicted'], -1)
                cv2.circle(frame, pixel_pos, self.dot_size + 2, self.colors['target_predicted'], 2)
    
    def draw_camera_aim(self, frame: np.ndarray):
        """Draw camera aim point indicator"""
        pixel_pos = self.normalize_to_pixel(self.camera_position)
        
        # Red diamond for camera aim
        diamond_size = self.camera_dot_size
        diamond_points = np.array([
            [pixel_pos[0], pixel_pos[1] - diamond_size],  # Top
            [pixel_pos[0] + diamond_size, pixel_pos[1]],   # Right
            [pixel_pos[0], pixel_pos[1] + diamond_size],   # Bottom
            [pixel_pos[0] - diamond_size, pixel_pos[1]]    # Left
        ], np.int32)
        
        cv2.fillPoly(frame, [diamond_points], self.colors['camera_aim'])
        cv2.polylines(frame, [diamond_points], True, (0, 0, 0), 2)
    
    def draw_info_panel(self, frame: np.ndarray, mode: str, confidence: float, 
                       frame_rate: float, detection_rate: float):
        """Draw information panel with tracking stats"""
        if self.frame_height is None:
            return
        
        # Background rectangle for text
        panel_height = 120
        panel_width = 300
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), self.colors['text'], 1)
        
        # Text information
        info_lines = [
            f"Mode: {mode}",
            f"Confidence: {confidence:.2f}",
            f"FPS: {frame_rate:.1f}",
            f"Detection: {detection_rate:.1f}%",
            f"Targets: {len(self.target_history)}",
        ]
        
        y_offset = 30
        for line in info_lines:
            cv2.putText(frame, line, (20, y_offset), self.font, 
                       self.font_scale, self.colors['text'], self.font_thickness)
            y_offset += 20
    
    def draw_prediction_vector(self, frame: np.ndarray, 
                             current_pos: Tuple[float, float], 
                             predicted_pos: Tuple[float, float]):
        """Draw prediction vector arrow"""
        current_pixel = self.normalize_to_pixel(current_pos)
        predicted_pixel = self.normalize_to_pixel(predicted_pos)
        
        # Draw arrow from current to predicted position
        cv2.arrowedLine(frame, current_pixel, predicted_pixel, 
                       self.colors['target_predicted'], 2, tipLength=0.3)
    
    def process_frame(self, frame: np.ndarray, 
                     target_coords: Optional[Tuple[float, float]] = None,
                     camera_coords: Optional[Tuple[float, float]] = None,
                     is_detected: bool = False,
                     mode: str = "UNKNOWN",
                     confidence: float = 0.0,
                     frame_rate: float = 0.0,
                     detection_rate: float = 0.0,
                     predicted_coords: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Process frame with tracking visualization overlay.
        
        Args:
            frame: Input frame to overlay graphics on
            target_coords: Current target coordinates (normalized 0-1)
            camera_coords: Current camera aim coordinates (normalized 0-1)
            is_detected: Whether target was detected or predicted
            mode: Current tracking mode ("PREDICTION", "EXTRAPOLATION", etc.)
            confidence: Detection/prediction confidence (0-1)
            frame_rate: Current frame processing rate
            detection_rate: Success rate percentage
            predicted_coords: Predicted future position (if different from target_coords)
        
        Returns:
            Frame with visualization overlay
        """
        # Set frame dimensions on first call
        if self.frame_height is None or self.frame_width is None:
            self.frame_height, self.frame_width = frame.shape[:2]
        
        # Create copy for overlay
        vis_frame = frame.copy()
        
        # Update positions
        if target_coords is not None:
            self.update_target_position(target_coords, is_detected)
        
        if camera_coords is not None:
            self.update_camera_position(camera_coords)
        
        # Draw visualization elements
        self.draw_crosshairs(vis_frame)
        self.draw_tracking_trail(vis_frame)
        
        # Draw current target
        if target_coords is not None:
            self.draw_target_indicators(vis_frame, target_coords, is_detected)
            
            # Draw prediction vector if we have both current and predicted positions
            if predicted_coords is not None and predicted_coords != target_coords:
                self.draw_prediction_vector(vis_frame, target_coords, predicted_coords)
        
        # Draw camera aim point
        self.draw_camera_aim(vis_frame)
        
        # Draw info panel
        self.draw_info_panel(vis_frame, mode, confidence, frame_rate, detection_rate)
        
        return vis_frame
    
    def clear_history(self):
        """Clear all tracking history"""
        self.target_history.clear()
        self.prediction_history.clear()
    
    def get_tracking_stats(self) -> dict:
        """Get current tracking statistics"""
        return {
            'target_history_length': len(self.target_history),
            'prediction_history_length': len(self.prediction_history),
            'camera_position': self.camera_position,
            'frame_size': (self.frame_width, self.frame_height) if self.frame_width else None
        }