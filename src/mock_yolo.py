import cv2
import numpy as np
from typing import Tuple, Optional, List
import random
import math

class MockYoloDetector:
    """
    Mock YOLO detector for testing the target tracking pipeline.
    Simulates realistic detection behavior including false positives, 
    missed detections, and varying confidence scores.
    """
    
    def __init__(self):
        self.confidence_threshold = 0.5
        self.target_classes = ["aircraft", "drone", "helicopter"]
        
        # Simulation parameters
        self.base_detection_rate = 0.85  # Base probability of detection
        self.false_positive_rate = 0.05  # Probability of false detection
        self.confidence_noise = 0.15  # Noise in confidence scores
        
        # Target tracking for consistency
        self.previous_detections = []
        self.detection_history = []
        
    def detect_targets(self, frame: np.ndarray) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Mock target detection that simulates realistic YOLO behavior
        Args:
            frame: Input frame from camera
        Returns:
            (target_coordinates, confidence_score) where coordinates are normalized [0,1]
        """
        height, width = frame.shape[:2]
        
        # Analyze frame to find potential targets (mock analysis)
        potential_targets = self._find_potential_targets(frame)
        
        if potential_targets:
            # Select best target based on mock criteria
            best_target = self._select_best_target(potential_targets, frame)
            
            # Add realistic detection noise and failures
            detection_result = self._apply_detection_realism(best_target, frame)
            
            return detection_result
        else:
            # No targets found, but might have false positive
            if random.random() < self.false_positive_rate:
                return self._generate_false_positive(width, height)
            else:
                return None, 0.0
                
    def _find_potential_targets(self, frame: np.ndarray) -> List[dict]:
        """Mock target detection based on simple image analysis"""
        height, width = frame.shape[:2]
        potential_targets = []
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Look for dark objects against light background (aircraft silhouettes)
        # This is a very simplified mock - real YOLO would use deep learning
        
        # Method 1: Find dark regions (aircraft against sky)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by size (reasonable aircraft/drone size)
            if 50 < area < 5000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center point
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Normalize coordinates
                norm_x = center_x / width
                norm_y = center_y / height
                
                # Mock confidence based on size and shape
                aspect_ratio = w / h if h > 0 else 1.0
                confidence = self._calculate_mock_confidence(area, aspect_ratio, center_x, center_y, width, height)
                
                potential_targets.append({
                    'coords': (norm_x, norm_y),
                    'confidence': confidence,
                    'area': area,
                    'bbox': (x, y, w, h)
                })
                
        # Method 2: Look for moving objects (mock motion detection)
        motion_targets = self._detect_motion_targets(frame)
        potential_targets.extend(motion_targets)
        
        return potential_targets
        
    def _detect_motion_targets(self, frame: np.ndarray) -> List[dict]:
        """Mock motion-based target detection"""
        height, width = frame.shape[:2]
        motion_targets = []
        
        # Simple mock motion detection - in reality this would use background subtraction
        # or optical flow. For now, we'll simulate finding targets in typical locations
        
        # Simulate aircraft/drone typical flight patterns
        typical_locations = [
            (0.3, 0.4),  # Left side, slightly above center
            (0.7, 0.3),  # Right side, upper third
            (0.5, 0.6),  # Center, lower third
            (0.2, 0.2),  # Upper left
            (0.8, 0.8),  # Lower right
        ]
        
        # Randomly activate some locations as "motion detections"
        for loc in typical_locations:
            if random.random() < 0.3:  # 30% chance each location has motion
                # Add some noise to the location
                noise_x = random.uniform(-0.05, 0.05)
                noise_y = random.uniform(-0.05, 0.05)
                
                final_x = max(0, min(1, loc[0] + noise_x))
                final_y = max(0, min(1, loc[1] + noise_y))
                
                confidence = random.uniform(0.4, 0.9)
                
                motion_targets.append({
                    'coords': (final_x, final_y),
                    'confidence': confidence,
                    'area': random.randint(100, 1000),
                    'bbox': (0, 0, 50, 50)  # Mock bbox
                })
                
        return motion_targets
        
    def _calculate_mock_confidence(self, area: float, aspect_ratio: float, x: int, y: int, 
                                 width: int, height: int) -> float:
        """Calculate mock confidence score based on target characteristics"""
        base_confidence = 0.5
        
        # Size factor - medium sized objects are more confident
        if 200 < area < 2000:
            size_bonus = 0.3
        elif 100 < area < 200 or 2000 < area < 3000:
            size_bonus = 0.1
        else:
            size_bonus = -0.1
            
        # Aspect ratio factor - aircraft-like shapes
        if 0.3 < aspect_ratio < 0.7 or 1.5 < aspect_ratio < 4.0:
            shape_bonus = 0.2
        else:
            shape_bonus = -0.1
            
        # Position factor - avoid edges where detection is less reliable
        edge_distance = min(x, y, width - x, height - y) / min(width, height)
        if edge_distance > 0.1:
            position_bonus = 0.1
        else:
            position_bonus = -0.2
            
        # Add some random noise
        noise = random.uniform(-self.confidence_noise, self.confidence_noise)
        
        final_confidence = base_confidence + size_bonus + shape_bonus + position_bonus + noise
        return max(0.0, min(1.0, final_confidence))
        
    def _select_best_target(self, targets: List[dict], frame: np.ndarray) -> dict:
        """Select the best target from potential detections"""
        if not targets:
            return None
            
        # Filter targets above confidence threshold
        valid_targets = [t for t in targets if t['confidence'] > self.confidence_threshold]
        
        if not valid_targets:
            return None
            
        # Select target with highest confidence
        best_target = max(valid_targets, key=lambda t: t['confidence'])
        
        # Apply temporal consistency - prefer targets near previous detections
        if self.previous_detections:
            for prev_detection in list(self.previous_detections)[-3:]:  # Last 3 detections
                prev_x, prev_y = prev_detection['coords']
                
                for target in valid_targets:
                    curr_x, curr_y = target['coords']
                    distance = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    
                    # Boost confidence for nearby targets (temporal consistency)
                    if distance < 0.1:  # Close to previous detection
                        target['confidence'] *= 1.2
                        
            # Re-select best target after temporal boosting
            best_target = max(valid_targets, key=lambda t: t['confidence'])
            
        return best_target
        
    def _apply_detection_realism(self, target: dict, frame: np.ndarray) -> Tuple[Optional[Tuple[float, float]], float]:
        """Apply realistic detection failures and noise"""
        if target is None:
            return None, 0.0
            
        coords = target['coords']
        confidence = target['confidence']
        
        # Simulate detection failures even for good targets
        detection_probability = min(0.95, self.base_detection_rate * confidence)
        
        if random.random() > detection_probability:
            # Detection failed
            return None, 0.0
            
        # Add coordinate noise (simulating detection jitter)
        noise_x = random.uniform(-0.02, 0.02)
        noise_y = random.uniform(-0.02, 0.02)
        
        noisy_x = max(0, min(1, coords[0] + noise_x))
        noisy_y = max(0, min(1, coords[1] + noise_y))
        
        # Add confidence noise
        confidence_noise = random.uniform(-0.1, 0.1)
        final_confidence = max(0.1, min(1.0, confidence + confidence_noise))
        
        # Store for temporal consistency
        detection_data = {
            'coords': (noisy_x, noisy_y),
            'confidence': final_confidence,
            'timestamp': cv2.getTickCount()
        }
        self.previous_detections.append(detection_data)
        
        # Keep only recent detections
        if len(self.previous_detections) > 10:
            self.previous_detections.pop(0)
            
        return (noisy_x, noisy_y), final_confidence
        
    def _generate_false_positive(self, width: int, height: int) -> Tuple[Tuple[float, float], float]:
        """Generate a false positive detection"""
        # Random location
        false_x = random.uniform(0.1, 0.9)
        false_y = random.uniform(0.1, 0.9)
        
        # Low to medium confidence for false positives
        false_confidence = random.uniform(0.3, 0.7)
        
        return (false_x, false_y), false_confidence
        
    def get_detection_statistics(self) -> dict:
        """Get statistics about mock detection performance"""
        if not self.detection_history:
            return {'total_detections': 0, 'avg_confidence': 0.0}
            
        confidences = [d['confidence'] for d in self.detection_history if d['confidence'] > 0]
        
        return {
            'total_detections': len(self.detection_history),
            'successful_detections': len(confidences),
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'detection_rate': len(confidences) / len(self.detection_history) if self.detection_history else 0.0
        }