import threading
import time
import cv2
import numpy as np
from typing import Optional, Tuple, List
from .frame_data import FrameData, ProcessedFrameData
from .queues import FrameQueue, ProcessedFrameQueue
from .mock_yolo import MockYoloDetector

class YoloProcessor(threading.Thread):
    """Thread for processing frames with YOLO model to detect targets"""
    
    def __init__(self, frame_queue: FrameQueue, processed_frame_queue: ProcessedFrameQueue, use_mock: bool = True):
        super().__init__(name="YoloProcessor")
        self.frame_queue = frame_queue
        self.processed_frame_queue = processed_frame_queue
        self.running = False
        self.use_mock = use_mock
        
        # YOLO model parameters (to be loaded from actual model)
        self.model = None
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.target_classes = ["person", "drone", "airplane"]  # Adjust based on your targets
        
        # Initialize mock detector if using mock mode
        if self.use_mock:
            self.mock_detector = MockYoloDetector()
        
    def start(self):
        """Start the YOLO processing thread"""
        self.running = True
        super().start()
        
    def stop(self):
        """Stop the YOLO processing thread"""
        self.running = False
        
    def load_model(self, model_path: str, config_path: str, classes_path: str):
        """Load YOLO model from files (placeholder for actual implementation)"""
        try:
            # Placeholder - replace with actual YOLO model loading
            # self.model = cv2.dnn.readNet(model_path, config_path)
            # with open(classes_path, 'r') as f:
            #     self.classes = [line.strip() for line in f.readlines()]
            print(f"Model loading placeholder - would load from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return False
            
    def detect_targets(self, frame: np.ndarray) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Process frame with YOLO to detect targets
        Returns: (target_coordinates, confidence_score)
        """
        try:
            if self.use_mock:
                # Use mock detector for testing
                return self.mock_detector.detect_targets(frame)
            else:
                # Actual YOLO inference (placeholder)
                # In real implementation, this would:
                # 1. Preprocess the frame for YOLO input
                # 2. Run inference
                # 3. Apply NMS to filter detections
                # 4. Extract highest confidence target of interest
                
                # Placeholder - replace with actual YOLO model
                return None, 0.0
                
        except Exception as e:
            print(f"Error in target detection: {e}")
            return None, 0.0
            
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO input"""
        # Typical YOLO preprocessing
        # Resize, normalize, convert BGR to RGB if needed
        processed_frame = cv2.resize(frame, (416, 416))  # Common YOLO input size
        return processed_frame
        
    def run(self):
        """Main thread execution - process frames from queue"""
        print("YOLO processor thread started")
        
        while self.running:
            try:
                # Get frame from input queue
                frame_data = self.frame_queue.get(timeout=0.1)
                
                # Preprocess frame
                processed_frame = self.preprocess_frame(frame_data.frame)
                
                # Detect targets
                target_coords, confidence = self.detect_targets(processed_frame)
                
                # Create processed frame data
                processed_data = ProcessedFrameData(
                    target_coords=target_coords,
                    processed_frame_label=frame_data.frame_label,
                    confidence=confidence,
                    timestamp=time.time()
                )
                
                # Add to processed queue
                try:
                    self.processed_frame_queue.put(processed_data, timeout=0.01)
                except:
                    # Queue full, drop frame - acceptable for real-time processing
                    pass
                    
            except KeyboardInterrupt:
                print("YOLO processor interrupted")
                break
            except Exception as e:
                if self.running:
                    # Only log errors if we're supposed to be running
                    error_msg = str(e).lower()
                    if "timed out" not in error_msg and "empty" not in error_msg:
                        print(f"Error in YOLO processing: {e}")
                        import traceback
                        traceback.print_exc()
                        break  # Exit on serious errors
                else:
                    break
                
        print("YOLO processor thread stopped")