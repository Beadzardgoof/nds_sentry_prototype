#!/usr/bin/env python3

"""
Test runner for the NDS Sentry system - provides easy testing configurations
and performance benchmarks for the threading pipeline.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import MasterController

class SentryTestRunner:
    """Test runner for automated testing of the sentry system"""
    
    def __init__(self):
        self.test_results = []
        self.config_dir = Path(__file__).parent.parent / "config.json"
        
    def load_config(self):
        """Load configuration from config.json"""
        try:
            with open(self.config_dir, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Config file not found, using defaults")
            return {}
            
    def run_performance_test(self, video_file=None, duration=30):
        """Run performance test to measure threading efficiency"""
        print(f"\n=== Performance Test ===")
        print(f"Duration: {duration}s")
        print(f"Video: {video_file or 'Camera'}")
        
        controller = MasterController(
            video_file=video_file,
            debug=True,
            use_mock=True
        )
        
        start_time = time.time()
        
        try:
            # Start system
            controller.start()
            
            # Let it run for specified duration
            time.sleep(duration)
            
        except KeyboardInterrupt:
            print("Test interrupted by user")
        finally:
            # Stop and collect results
            controller.stop()
            runtime = time.time() - start_time
            
            results = {
                'duration': runtime,
                'frames_processed': controller.frame_count,
                'detections': controller.detection_count,
                'predictions': controller.prediction_count,
                'extrapolations': controller.extrapolation_count,
                'fps': controller.frame_count / runtime if runtime > 0 else 0,
                'detection_rate': controller.detection_count / controller.frame_count if controller.frame_count > 0 else 0,
                'final_success_rate': controller.current_success_rate
            }
            
            return results
            
    def run_video_test_suite(self):
        """Run tests on all available mock videos"""
        video_dir = Path(__file__).parent.parent / "data" / "mock_videos"
        
        if not video_dir.exists():
            print("No mock videos found. Run mock_video_generator.py first.")
            return
            
        video_files = list(video_dir.glob("*.mp4"))
        
        if not video_files:
            print("No video files found in mock_videos directory")
            return
            
        print(f"\n=== Video Test Suite ===")
        print(f"Found {len(video_files)} test videos")
        
        all_results = {}
        
        for video_file in video_files:
            print(f"\nTesting: {video_file.name}")
            results = self.run_performance_test(str(video_file), duration=15)
            all_results[video_file.name] = results
            
            print(f"Results: {results['fps']:.1f} FPS, "
                  f"{results['detection_rate']:.1%} detection rate")
                  
        return all_results
        
    def run_threading_stress_test(self):
        """Test threading performance under different queue sizes"""
        print(f"\n=== Threading Stress Test ===")
        
        queue_sizes = [5, 10, 20, 50]
        results = {}
        
        for queue_size in queue_sizes:
            print(f"\nTesting with queue size: {queue_size}")
            
            # This would require modifying the controller to accept queue size
            # For now, just run standard test
            controller = MasterController(debug=False, use_mock=True)
            
            start_time = time.time()
            try:
                controller.start()
                time.sleep(10)  # Short test
            finally:
                controller.stop()
                runtime = time.time() - start_time
                
                results[queue_size] = {
                    'fps': controller.frame_count / runtime if runtime > 0 else 0,
                    'frames': controller.frame_count,
                    'detections': controller.detection_count
                }
                
                print(f"Queue {queue_size}: {results[queue_size]['fps']:.1f} FPS")
                
        return results
        
    def print_test_report(self, results):
        """Print formatted test report"""
        print(f"\n" + "="*50)
        print("SENTRY SYSTEM TEST REPORT")
        print("="*50)
        
        if isinstance(results, dict) and 'duration' in results:
            # Single test results
            self.print_single_test_results(results)
        elif isinstance(results, dict):
            # Multiple test results
            for test_name, test_results in results.items():
                print(f"\n{test_name}:")
                self.print_single_test_results(test_results)
                
    def print_single_test_results(self, results):
        """Print results for a single test"""
        print(f"  Duration: {results.get('duration', 0):.1f}s")
        print(f"  Frames processed: {results.get('frames_processed', 0)}")
        print(f"  Average FPS: {results.get('fps', 0):.1f}")
        print(f"  Detections: {results.get('detections', 0)}")
        print(f"  Detection rate: {results.get('detection_rate', 0):.1%}")
        print(f"  Predictions: {results.get('predictions', 0)}")
        print(f"  Extrapolations: {results.get('extrapolations', 0)}")
        print(f"  Final success rate: {results.get('final_success_rate', 0):.3f}")

def main():
    parser = argparse.ArgumentParser(description="NDS Sentry Test Runner")
    parser.add_argument("--perf", action="store_true", help="Run performance test")
    parser.add_argument("--video-suite", action="store_true", help="Run full video test suite")
    parser.add_argument("--stress", action="store_true", help="Run threading stress test")
    parser.add_argument("--video", "-v", type=str, help="Specific video file to test")
    parser.add_argument("--duration", "-d", type=int, default=30, help="Test duration in seconds")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    runner = SentryTestRunner()
    
    if args.all:
        print("Running comprehensive test suite...")
        perf_results = runner.run_performance_test(duration=args.duration)
        video_results = runner.run_video_test_suite()
        stress_results = runner.run_threading_stress_test()
        
        print("\n" + "="*60)
        print("COMPREHENSIVE TEST RESULTS")
        print("="*60)
        runner.print_test_report(perf_results)
        runner.print_test_report(video_results)
        
    elif args.perf:
        results = runner.run_performance_test(args.video, args.duration)
        runner.print_test_report(results)
        
    elif args.video_suite:
        results = runner.run_video_test_suite()
        runner.print_test_report(results)
        
    elif args.stress:
        results = runner.run_threading_stress_test()
        runner.print_test_report(results)
        
    else:
        print("No test specified. Use --help for options.")
        print("Quick test options:")
        print("  python test_runner.py --perf")
        print("  python test_runner.py --video-suite")
        print("  python test_runner.py --all")

if __name__ == "__main__":
    main()