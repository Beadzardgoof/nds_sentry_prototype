#!/usr/bin/env python3

import threading
import time
import argparse
import os
import signal
import sys
import queue
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
from src.thread_pool_manager import ThreadPoolManager, ThreadType

class MasterController:
    def __init__(self, video_file=None, debug=False, use_mock=True):
        # Initialize queues
        self.frame_queue = FrameQueue(maxsize=10)
        self.processed_frame_queue = ProcessedFrameQueue(maxsize=10)
        
        # Debug and configuration options
        self.debug = debug
        self.use_mock = use_mock
        
        # Initialize thread pool manager (temporarily disabled for basic functionality)
        # self.thread_pool = ThreadPoolManager(max_threads=20)
        self.thread_pool = None
        
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
        self.mode = "prediction"  # Track current mode: "prediction" or "extrapolation"
        
        # Debug statistics
        self.frame_count = 0
        self.detection_count = 0
        self.prediction_count = 0
        self.extrapolation_count = 0
        self.scan_pattern_count = 0
        self.start_time = None
        
    def start(self, max_runtime=None):
        """Start all processing threads and main control loop"""
        self.running = True
        self.start_time = time.time()
        self.max_runtime = max_runtime  # Optional timeout for testing
        
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
            if max_runtime:
                print(f"Max runtime: {max_runtime}s")
            print("Starting threads...")
        
        try:
            # Start capture and processing threads
            self.image_capture.start()
            self.yolo_processor.start()
            
            if self.debug:
                print("Threads started. Beginning control loop...")
            
            # Start main control loop
            self.control_loop()
        except Exception as e:
            if self.debug:
                print(f"Error in start(): {e}")
            raise
        finally:
            # Ensure cleanup happens
            self.stop()
        
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
        last_rebalance_time = time.time()
        debug_interval = 2.0  # Print debug info every 2 seconds
        rebalance_interval = 0.1  # Rebalance threads every 100ms
        
        while self.running:
            try:
                # Check for timeout if specified
                if self.max_runtime and (time.time() - self.start_time) > self.max_runtime:
                    if self.debug:
                        print(f"\nMax runtime ({self.max_runtime}s) reached, stopping...")
                    break
                
                # Get processed frame data with timeout
                processed_data = self.processed_frame_queue.get(timeout=0.1)
                self.frame_count += 1
                
                # Check frame freshness
                if not processed_data.is_fresh(max_age=0.1):
                    if self.debug:
                        print("Stale frame detected, skipping")
                    continue
                
                # Update thread pool with success/failure information
                has_detection = processed_data.target_coords is not None
                if self.thread_pool:
                    self.thread_pool.update_failure_rate(has_detection)
                
                if has_detection:
                    self.detection_count += 1
                    
                    # Update success rate based on detection confidence
                    self.update_success_rate(processed_data.confidence)
                    
                    # Check if we need to switch modes based on failure rate prediction
                    if self.thread_pool:
                        if self.thread_pool.should_switch_to_extrapolation() and self.mode != "extrapolation":
                            self.mode = "extrapolation"
                            if self.debug:
                                print("Switching to EXTRAPOLATION mode")
                        elif self.thread_pool.should_switch_to_prediction() and self.mode != "prediction":
                            self.mode = "prediction"
                            if self.debug:
                                print("Switching to PREDICTION mode")
                    
                    if self.mode == "prediction":
                        # Use prediction model for coordinates
                        predicted_coords = self.target_prediction.predict(
                            processed_data.target_coords, 
                            processed_data.processed_frame_label
                        )
                        self.servo_controller.move_to_target(predicted_coords)
                        self.prediction_count += 1
                        
                        if self.debug:
                            self.print_debug_info(processed_data, "PREDICTION", predicted_coords)
                    else:
                        # Use extrapolation model for scan patterns
                        if processed_data.has_scan_pattern():
                            # Execute scan pattern
                            self.servo_controller.execute_scan_pattern(processed_data.scan_pattern)
                            self.scan_pattern_count += 1
                            
                            if self.debug:
                                pattern = processed_data.scan_pattern
                                self.print_debug_info(processed_data, f"SCAN_{pattern.pattern_type.value.upper()}", 
                                                    pattern.center_coords)
                        else:
                            # Fallback to coordinate extrapolation
                            extrapolated_coords = self.target_extrapolation.extrapolate(
                                processed_data.target_coords,
                                processed_data.processed_frame_label
                            )
                            self.servo_controller.move_to_target(extrapolated_coords)
                            self.extrapolation_count += 1
                            
                            if self.debug:
                                self.print_debug_info(processed_data, "EXTRAPOLATION", extrapolated_coords)
                else:
                    # No detection - generate scan pattern or fallback
                    if processed_data.has_scan_pattern():
                        # Execute scan pattern from extrapolation thread
                        self.servo_controller.execute_scan_pattern(processed_data.scan_pattern)
                        self.scan_pattern_count += 1
                        
                        if self.debug:
                            pattern = processed_data.scan_pattern
                            self.print_debug_info(processed_data, f"SEARCH_{pattern.pattern_type.value.upper()}", 
                                                pattern.center_coords)
                    else:
                        # Fallback to coordinate extrapolation
                        extrapolated_coords = self.target_extrapolation.extrapolate(
                            None, processed_data.processed_frame_label
                        )
                        self.servo_controller.move_to_target(extrapolated_coords)
                        self.extrapolation_count += 1
                        
                        if self.debug:
                            self.print_debug_info(processed_data, "NO_DETECTION", extrapolated_coords)
                
                # Rebalance thread pool periodically
                if self.thread_pool and time.time() - last_rebalance_time > rebalance_interval:
                    self.thread_pool.rebalance_threads()
                    last_rebalance_time = time.time()
                
                # Print periodic debug summary
                if self.debug and time.time() - last_debug_time > debug_interval:
                    self.print_debug_summary()
                    last_debug_time = time.time()
                
            except KeyboardInterrupt:
                print("Control loop interrupted")
                break
            except queue.Empty:
                # Normal timeout when no frames available
                continue
            except Exception as e:
                # Handle timeout or other exceptions
                if not self.running:
                    break
                    
                error_msg = str(e).lower()
                # These are normal queue timeout/empty conditions - don't log them
                if any(term in error_msg for term in ["timed out", "empty", "timeout"]):
                    continue
                    
                # Log unexpected errors only
                if self.debug:
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
        
        # Get thread pool status
        if self.thread_pool:
            thread_status = self.thread_pool.get_status()
        else:
            thread_status = {'failure_rate': 0.0, 'prediction_threshold': False, 'total_threads': 2, 'max_threads': 20, 'allocations': {}}
        
        print(f"\n=== DEBUG SUMMARY [{runtime:.1f}s] ===")
        print(f"Mode: {self.mode.upper()}")
        print(f"Frames processed: {self.frame_count} ({fps:.1f} FPS)")
        print(f"Detections: {self.detection_count} ({detection_rate:.1%})")
        print(f"Predictions: {self.prediction_count}")
        print(f"Extrapolations: {self.extrapolation_count}")
        print(f"Scan patterns: {self.scan_pattern_count}")
        print(f"Current success rate: {self.current_success_rate:.3f}")
        print(f"Thread pool failure rate: {thread_status['failure_rate']:.3f}")
        print(f"Prediction threshold triggered: {thread_status['prediction_threshold']}")
        print(f"Frame queue size: {self.frame_queue.qsize()}")
        print(f"Processed queue size: {self.processed_frame_queue.qsize()}")
        print(f"Active threads: {thread_status['total_threads']}/{thread_status['max_threads']}")
        
        # Show thread allocation
        allocations = thread_status['allocations']
        for thread_type, count in allocations.items():
            if count > 0:
                print(f"  {thread_type}: {count}")
        
        print("=" * 50)

    def get_thread_pool_status(self):
        """Get thread pool status for testing"""
        if self.thread_pool:
            return self.thread_pool.get_status()
        else:
            return {'failure_rate': 0.0, 'prediction_threshold': False, 'total_threads': 2, 'max_threads': 20, 'allocations': {}}

    def stop(self):
        """Stop all processing threads"""
        self.running = False
        
        if self.debug:
            print("Stopping threads...")
        
        # Stop thread pool
        if hasattr(self, 'thread_pool') and self.thread_pool:
            self.thread_pool.shutdown()
        
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
        controller.running = False  # Stop control loop
        if args.debug:
            print("\n=== FINAL STATISTICS ===")
            controller.print_debug_summary()
        controller.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):  # SIGTERM not available on Windows
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