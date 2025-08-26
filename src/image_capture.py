import threading
import time
import cv2
import numpy as np
from typing import Optional
from .frame_data import FrameData
from .queues import FrameQueue

class ImageCaptureThread(threading.Thread):
    """Thread for capturing camera frames and adding them to the frame queue"""
    
    def __init__(self, frame_queue: FrameQueue, camera_index: int = 0, video_file: str = None):
        super().__init__(name="ImageCapture")
        self.frame_queue = frame_queue
        self.camera_index = camera_index
        self.video_file = video_file
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
        if self.camera is not None:
            self.camera.release()
            
    def initialize_camera(self) -> bool:
        """Initialize the camera or video file capture"""
        try:
            if self.is_video_file:
                # Open video file
                self.camera = cv2.VideoCapture(self.video_file)
                if not self.camera.isOpened():
                    print(f"Error: Could not open video file {self.video_file}")
                    return False
                    
                # Get video properties
                self.fps_target = int(self.camera.get(cv2.CAP_PROP_FPS)) or 30
                frame_count = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / self.fps_target
                print(f"Loaded video: {self.video_file}")
                print(f"Video properties: {frame_count} frames, {self.fps_target} FPS, {duration:.1f}s duration")
                
            else:
                # Open camera
                self.camera = cv2.VideoCapture(self.camera_index)
                if not self.camera.isOpened():
                    print(f"Error: Could not open camera {self.camera_index}")
                    return False
                    
                # Set camera properties for optimal performance
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.camera.set(cv2.CAP_PROP_FPS, self.fps_target)
                
                print(f"Initialized camera {self.camera_index}")
            
            return True
        except Exception as e:
            print(f"Error initializing camera/video: {e}")
            return False
            
    def run(self):
        """Main thread execution - capture frames and add to queue"""
        if not self.initialize_camera():
            print("Failed to initialize camera, exiting capture thread")
            return
            
        frame_interval = 1.0 / self.fps_target
        last_frame_time = 0
        
        while self.running:
            current_time = time.time()
            
            # Maintain target FPS
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
                
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
            
        # Cleanup
        if self.camera is not None:
            self.camera.release()
            print("Camera released")