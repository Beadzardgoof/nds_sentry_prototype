#!/usr/bin/env python3

import threading
import time
import argparse
import os
import signal
import sys
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from src.image_capture import ImageCaptureThread
from src.yolo_processor import YoloProcessor
from src.target_prediction import TargetPredictionModel
from src.target_extrapolation import TargetExtrapolationModel
from src.servo_controller import ServoController
from src.frame_data import FrameData, ProcessedFrameData
from src.queues import FrameQueue, ProcessedFrameQueue

class MasterController:
    def __init__(self, video_file=None, debug=False, use_mock=True):
        # Initialize queues
        self.frame_queue = FrameQueue(maxsize=10)
        self.processed_frame_queue = ProcessedFrameQueue(maxsize=10)
        
        # Debug and configuration options
        self.debug = debug
        self.use_mock = use_mock
        
        # Initialize processing components
        # Use synthetic frames if no video file specified (for testing without camera)
        use_synthetic = video_file is None
        self.image_capture = ImageCaptureThread(self.frame_queue, video_file=video_file, use_synthetic=use_synthetic)
        self.yolo_processor = YoloProcessor(self.frame_queue, self.processed_frame_queue, use_mock=use_mock)
        self.target_prediction = TargetPredictionModel()
        self.target_extrapolation = TargetExtrapolationModel()
        self.servo_controller = ServoController()
        
        # Control parameters
        self.success_rate_threshold = 0.8
        self.current_success_rate = 0.0
        self.running = False
        
        # Debug statistics
        self.frame_count = 0
        self.detection_count = 0
        self.prediction_count = 0
        self.extrapolation_count = 0
        self.start_time = None
        
    def start(self):
        """Start all processing threads and main control loop"""
        self.running = True
        self.start_time = time.time()
        
        if self.debug:
            print("=== NDS Sentry Debug Mode ===")
            print(f"Mock detection: {self.use_mock}")
            video_source = getattr(self.image_capture, 'video_file', None)
            if video_source:
                print(f"Video file: {video_source}")
            elif getattr(self.image_capture, 'use_synthetic', False):
                print("Video source: Synthetic frames (no camera)")
            else:
                print("Video source: Camera")
            print("Starting threads...")
        
        # Start capture and processing threads
        self.image_capture.start()
        self.yolo_processor.start()
        
        if self.debug:
            print("Threads started. Beginning control loop...")
        
        # Start main control loop
        self.control_loop()
        
    def stop(self):
        """Stop all processing threads"""
        self.running = False
        
        if self.debug:
            print("Stopping threads...")
        
        # Stop threads
        if hasattr(self, 'image_capture'):
            self.image_capture.stop()
        if hasattr(self, 'yolo_processor'):
            self.yolo_processor.stop()
            
        # Wait for threads to finish with timeout
        try:
            if hasattr(self, 'image_capture') and self.image_capture.is_alive():
                self.image_capture.join(timeout=2.0)
            if hasattr(self, 'yolo_processor') and self.yolo_processor.is_alive():
                self.yolo_processor.join(timeout=2.0)
        except Exception as e:
            if self.debug:
                print(f"Thread join error: {e}")
                
        if self.debug:
            print("All threads stopped")
        
    def control_loop(self):
        """Main control loop that processes target data and controls servos"""
        last_debug_time = time.time()
        debug_interval = 2.0  # Print debug info every 2 seconds
        
        while self.running:
            try:
                # Get processed frame data with timeout
                processed_data = self.processed_frame_queue.get(timeout=0.1)
                self.frame_count += 1
                
                if processed_data.target_coords is not None:
                    self.detection_count += 1
                    
                    # Update success rate based on detection confidence
                    self.update_success_rate(processed_data.confidence)
                    
                    if self.current_success_rate >= self.success_rate_threshold:
                        # High success rate - use target prediction model
                        predicted_coords = self.target_prediction.predict(
                            processed_data.target_coords, 
                            processed_data.frame_label
                        )
                        self.servo_controller.move_to_target(predicted_coords)
                        self.prediction_count += 1
                        
                        if self.debug:
                            self.print_debug_info(processed_data, "PREDICTION", predicted_coords)
                    else:
                        # Low success rate - use extrapolation model as fallback
                        extrapolated_coords = self.target_extrapolation.extrapolate(
                            processed_data.target_coords,
                            processed_data.frame_label
                        )
                        self.servo_controller.move_to_target(extrapolated_coords)
                        self.extrapolation_count += 1
                        
                        if self.debug:
                            self.print_debug_info(processed_data, "EXTRAPOLATION", extrapolated_coords)
                else:
                    # No detection - use extrapolation model
                    extrapolated_coords = self.target_extrapolation.extrapolate(
                        None, processed_data.processed_frame_label
                    )
                    self.servo_controller.move_to_target(extrapolated_coords)
                    self.extrapolation_count += 1
                    
                    if self.debug:
                        self.print_debug_info(processed_data, "NO_DETECTION", extrapolated_coords)
                
                # Print periodic debug summary
                if self.debug and time.time() - last_debug_time > debug_interval:
                    self.print_debug_summary()
                    last_debug_time = time.time()
                
            except KeyboardInterrupt:
                print("Control loop interrupted")
                break
            except Exception as e:
                # Handle timeout or other exceptions
                if not self.running:
                    break
                error_msg = str(e).lower()
                if "timed out" not in error_msg and "empty" not in error_msg and self.debug:
                    print(f"Control loop exception: {e}")
                    import traceback
                    traceback.print_exc()
                continue
                
    def update_success_rate(self, confidence: float):
        """Update running success rate based on detection confidence"""
        # Simple exponential moving average
        alpha = 0.1
        self.current_success_rate = (alpha * confidence + 
                                   (1 - alpha) * self.current_success_rate)
        
    def print_debug_info(self, processed_data, mode, final_coords):
        """Print detailed debug information for each frame"""
        current_time = time.time()
        runtime = current_time - self.start_time if self.start_time else 0
        
        if processed_data.target_coords is not None:
            print(f"[{runtime:.1f}s] {mode}: "
                  f"Raw=({processed_data.target_coords[0]:.3f},{processed_data.target_coords[1]:.3f}) "
                  f"Final=({final_coords[0]:.3f},{final_coords[1]:.3f}) "
                  f"Conf={processed_data.confidence:.3f} "
                  f"SuccessRate={self.current_success_rate:.3f}")
        else:
            print(f"[{runtime:.1f}s] {mode}: "
                  f"Raw=None "
                  f"Final=({final_coords[0]:.3f},{final_coords[1]:.3f}) "
                  f"Conf={processed_data.confidence:.3f} "
                  f"SuccessRate={self.current_success_rate:.3f}")
        
    def print_debug_summary(self):
        """Print periodic debug summary"""
        runtime = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / runtime if runtime > 0 else 0
        detection_rate = self.detection_count / self.frame_count if self.frame_count > 0 else 0
        
        print(f"\n=== DEBUG SUMMARY [{runtime:.1f}s] ===")
        print(f"Frames processed: {self.frame_count} ({fps:.1f} FPS)")
        print(f"Detections: {self.detection_count} ({detection_rate:.1%})")
        print(f"Predictions: {self.prediction_count}")
        print(f"Extrapolations: {self.extrapolation_count}")
        print(f"Current success rate: {self.current_success_rate:.3f}")
        print(f"Frame queue size: {self.frame_queue.qsize()}")
        print(f"Processed queue size: {self.processed_frame_queue.qsize()}")
        print("=" * 40)

def main():
    parser = argparse.ArgumentParser(description="NDS Sentry Target Tracking System")
    parser.add_argument("--video", "-v", type=str, help="Path to video file for testing")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug output")
    parser.add_argument("--no-mock", action="store_true", help="Disable mock YOLO (use real model)")
    parser.add_argument("--list-videos", "-l", action="store_true", help="List available test videos")
    
    args = parser.parse_args()
    
    # List available test videos
    if args.list_videos:
        video_dir = "data/mock_videos"
        if os.path.exists(video_dir):
            print("Available test videos:")
            for file in os.listdir(video_dir):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    print(f"  {file}")
        else:
            print("No test videos directory found. Run testing/mock_video_generator.py first.")
        return
    
    # Initialize controller
    use_mock = not args.no_mock
    controller = MasterController(
        video_file=args.video,
        debug=args.debug,
        use_mock=use_mock
    )
    
    # Set up signal handler for clean shutdown
    def signal_handler(signum, frame):
        print("\nShutdown signal received...")
        controller.stop()
        if args.debug:
            print("\n=== FINAL STATISTICS ===")
            controller.print_debug_summary()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        controller.start()
    except KeyboardInterrupt:
        signal_handler(None, None)
    except Exception as e:
        print(f"Unexpected error: {e}")
        controller.stop()
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()