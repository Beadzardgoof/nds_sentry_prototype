#!/usr/bin/env python3

"""
Basic test script to verify the threading system works without errors
"""

import time
import sys
from main import MasterController

def basic_test():
    print("Running basic threading test with synthetic frames...")
    
    controller = MasterController(
        video_file=None,  # Will use synthetic frames
        debug=True,
        use_mock=True
    )
    
    try:
        print("Starting controller...")
        # Run controller for 5 seconds
        import threading
        controller_thread = threading.Thread(target=controller.start)
        controller_thread.daemon = True
        controller_thread.start()
        
        # Let it run for 5 seconds
        time.sleep(5)
        
        print("Stopping controller...")
        controller.stop()
        
        print("Test completed successfully!")
        print(f"Frames processed: {controller.frame_count}")
        print(f"Detections: {controller.detection_count}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        controller.stop()
        return False

if __name__ == "__main__":
    success = basic_test()
    sys.exit(0 if success else 1)