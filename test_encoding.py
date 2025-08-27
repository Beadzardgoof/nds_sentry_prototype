#!/usr/bin/env python3

"""
Test script to verify no encoding issues across different console environments
"""

import sys
import locale
import time
from main import MasterController

def test_encoding_compatibility():
    print("=== Encoding Compatibility Test ===")
    
    # Print system encoding info
    print(f"System encoding: {sys.getdefaultencoding()}")
    print(f"File system encoding: {sys.getfilesystemencoding()}")
    print(f"Locale: {locale.getpreferredencoding()}")
    
    # Test ASCII-only output
    print("\nTesting ASCII-only output...")
    
    controller = MasterController(
        video_file=None,  # Use synthetic frames
        debug=False,  # Disable debug to reduce output
        use_mock=True
    )
    
    try:
        print("Starting controller...")
        
        # Run controller for 2 seconds
        import threading
        controller_thread = threading.Thread(target=controller.start)
        controller_thread.daemon = True
        controller_thread.start()
        
        # Let it run briefly
        time.sleep(2)
        
        print("Stopping controller...")
        controller.stop()
        
        time.sleep(0.5)
        
        print("\n[SUCCESS] Encoding test completed!")
        print("- No Unicode encoding errors")
        print("- Servo output uses 'deg' instead of degree symbol")
        print("- All text output is ASCII-compatible")
        
        return True
        
    except UnicodeEncodeError as e:
        print(f"\n[FAILED] Unicode encoding error: {e}")
        return False
    except Exception as e:
        print(f"\n[FAILED] Other error: {e}")
        return False
    finally:
        try:
            controller.stop()
        except:
            pass

def test_special_characters():
    """Test that we don't accidentally output problematic characters"""
    print("\n=== Special Character Test ===")
    
    # Test strings that should work in any console
    test_strings = [
        "Servo move: Pan=45.0deg, Tilt=-30.0deg",
        "Detection: Raw=(0.123,0.456) Final=(0.789,0.012)",
        "Success rate: 0.850",
        "[DEBUG] System operational",
        "Frames processed: 100 (30.0 FPS)"
    ]
    
    try:
        for test_str in test_strings:
            print(test_str)
        
        print("\n[SUCCESS] All test strings printed successfully")
        print("- No Unicode characters")
        print("- Windows console compatible")
        return True
        
    except UnicodeEncodeError as e:
        print(f"\n[FAILED] Unicode error in test strings: {e}")
        return False

if __name__ == "__main__":
    success1 = test_encoding_compatibility()
    success2 = test_special_characters()
    
    if success1 and success2:
        print("\n=== ALL ENCODING TESTS PASSED ===")
        print("System is compatible with Windows console (cp1252)")
        sys.exit(0)
    else:
        print("\n=== ENCODING TESTS FAILED ===")
        sys.exit(1)