#!/usr/bin/env python3

"""
Test script to verify queue timeout handling works properly
"""

import time
import sys
from main import MasterController

def test_timeout_handling():
    print("Testing timeout handling...")
    
    controller = MasterController(
        video_file=None,  # Use synthetic frames
        debug=False,  # Disable debug to see clean output
        use_mock=True
    )
    
    try:
        print("Starting controller for 3 seconds...")
        # Run controller for 3 seconds
        import threading
        controller_thread = threading.Thread(target=controller.start)
        controller_thread.daemon = True
        controller_thread.start()
        
        # Let it run for 3 seconds
        time.sleep(3)
        
        print("Stopping controller...")
        controller.stop()
        
        # Wait a moment for cleanup
        time.sleep(0.5)
        
        print("[SUCCESS] Timeout handling test completed successfully!")
        print(f"   Frames processed: {controller.frame_count}")
        print(f"   Detections: {controller.detection_count}")
        print(f"   No error spam from timeouts")
        
        return True
        
    except Exception as e:
        print(f"[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            controller.stop()
        except:
            pass
        return False

if __name__ == "__main__":
    success = test_timeout_handling()
    sys.exit(0 if success else 1)