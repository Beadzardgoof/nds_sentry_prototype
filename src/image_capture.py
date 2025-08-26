import threading
import time
import cv2
import numpy as np
from typing import Optional
from .frame_data import FrameData
from .queues import FrameQueue

class ImageCaptureThread(threading.Thread):
    """Thread for capturing camera frames and adding them to the frame queue"""
    
    def __init__(self, frame_queue: FrameQueue, camera_index: int = 0, video_file: str = None, use_synthetic: bool = False):
        super().__init__(name="ImageCapture")
        self.frame_queue = frame_queue
        self.camera_index = camera_index
        self.video_file = video_file
        self.use_synthetic = use_synthetic
        self.running = False
        self.camera = None
        self.fps_target = 30  # Target FPS for frame capture
        self.frame_counter = 0
        self.is_video_file = video_file is not None
        
    def start(self):
        """Start the image capture thread"""
        self.running = True
        super().start()
        
    def stop(self):
        """Stop the image capture thread"""
        self.running = False
        # Don't release camera here - do it in the thread cleanup
            
    def initialize_camera(self) -> bool:
        """Initialize the camera, video file, or synthetic frame generator"""
        try:
            if self.use_synthetic:
                print("Using synthetic frame generator (no camera/video required)")
                return True
            elif self.is_video_file:
                # Open video file
                self.camera = cv2.VideoCapture(self.video_file)
                if not self.camera.isOpened():
                    print(f"Error: Could not open video file {self.video_file}")
                    print("Falling back to synthetic frame generation")
                    self.use_synthetic = True
                    return True
                    
                # Get video properties
                self.fps_target = int(self.camera.get(cv2.CAP_PROP_FPS)) or 30
                frame_count = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / self.fps_target
                print(f"Loaded video: {self.video_file}")
                print(f"Video properties: {frame_count} frames, {self.fps_target} FPS, {duration:.1f}s duration")
                
            else:
                # Try to open camera
                self.camera = cv2.VideoCapture(self.camera_index)
                if not self.camera.isOpened():
                    print(f"Warning: Could not open camera {self.camera_index}")
                    print("Falling back to synthetic frame generation for testing")
                    self.use_synthetic = True
                    return True
                    
                # Set camera properties for optimal performance
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.camera.set(cv2.CAP_PROP_FPS, self.fps_target)
                
                print(f"Initialized camera {self.camera_index}")
            
            return True
        except Exception as e:
            print(f"Error initializing camera/video: {e}")
            print("Falling back to synthetic frame generation")
            self.use_synthetic = True
            return True
            
    def run(self):
        """Main thread execution - capture frames and add to queue"""
        if not self.initialize_camera():
            print("Failed to initialize camera, exiting capture thread")
            return
            
        frame_interval = 1.0 / self.fps_target
        last_frame_time = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Maintain target FPS
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    continue
                
                if self.use_synthetic:
                    # Generate synthetic frame
                    frame = self.generate_synthetic_frame(current_time)
                    ret = True
                else:
                    # Read from camera/video
                    ret, frame = self.camera.read()
                    if not ret:
                        if self.is_video_file:
                            print("End of video file reached")
                            break  # End of video file
                        else:
                            print("Warning: Failed to capture frame")
                            continue
                
                # Create frame data with timestamp label
                frame_label = current_time
                frame_data = FrameData(
                    frame=frame,
                    frame_label=frame_label,
                    timestamp=current_time
                )
                
                # Add frame to queue (non-blocking to maintain real-time performance)
                try:
                    self.frame_queue.put(frame_data, timeout=0.01)
                    self.frame_counter += 1
                except:
                    # Queue full, frame dropped - this is acceptable for real-time processing
                    pass
                    
                last_frame_time = current_time
                
            except KeyboardInterrupt:
                print("Image capture interrupted")
                break
            except Exception as e:
                if self.running:
                    print(f"Error in image capture: {e}")
                break
            
        # Cleanup
        if self.camera is not None:
            self.camera.release()
            print("Camera released")
            
    def generate_synthetic_frame(self, timestamp: float) -> np.ndarray:
        """Generate a synthetic frame with moving targets for testing"""
        import math
        
        # Create frame
        height, width = 720, 1280
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create sky gradient background
        for y in range(height):
            intensity = int(200 - (y / height) * 50)  # Sky gradient
            frame[y, :] = [intensity, intensity - 10, 220]  # Blueish sky
        
        # Add some noise for realism
        noise = np.random.randint(0, 20, (height, width, 3))
        frame = cv2.add(frame, noise.astype(np.uint8))
        
        # Add moving targets
        time_factor = timestamp * 0.5  # Slow down movement
        
        # Target 1: Horizontal moving object (simulated aircraft)
        target1_x = int((width + 200) * ((time_factor % 6.0) / 6.0) - 100)
        target1_y = int(height * 0.3 + 50 * math.sin(time_factor * 2))
        
        if 0 <= target1_x < width and 0 <= target1_y < height:
            # Draw simple aircraft shape
            cv2.rectangle(frame, (target1_x-15, target1_y-3), (target1_x+15, target1_y+3), (100, 100, 100), -1)
            cv2.rectangle(frame, (target1_x-5, target1_y-12), (target1_x+5, target1_y+12), (80, 80, 80), -1)
        
        # Target 2: Circular motion (simulated drone)
        radius = 150
        center_x, center_y = width // 2, height // 2
        target2_x = int(center_x + radius * math.cos(time_factor))
        target2_y = int(center_y + radius * math.sin(time_factor))
        
        if 0 <= target2_x < width and 0 <= target2_y < height:
            # Draw simple drone shape
            cv2.circle(frame, (target2_x, target2_y), 8, (60, 60, 60), -1)
            # Rotors
            for dx, dy in [(-12, -12), (12, -12), (-12, 12), (12, 12)]:
                cv2.circle(frame, (target2_x + dx, target2_y + dy), 4, (40, 40, 40), -1)
        
        return frame