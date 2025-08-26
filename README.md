# NDS Sentry Prototype

A modular target tracking system for a two-servo sentry system. This prototype implements a complete threading pipeline for real-time target detection, prediction, and servo control.

## Architecture

### Threading Pipeline
```
Image Capture → Frame Queue → YOLO Processor → Processed Frame Queue → Master Controller
     ↓              ↓              ↓                    ↓                      ↓
  (frame,        stores         (target_coords,     stores target      Decision Logic:
frame_label)   raw frames     processed_frame_label,  data with       success_rate >= 0.8?
                               confidence)           timestamps       ├─ YES → Target Prediction
                                                                     └─ NO  → Target Extrapolation
                                                                             ↓
                                                                        Servo Controller
```

### Key Components

- **Master Controller**: Coordinates the entire system and makes targeting decisions
- **Image Capture Thread**: Captures frames from camera/video with temporal labeling
- **YOLO Processor Thread**: Detects targets with confidence scoring
- **Target Prediction Model**: High-accuracy prediction for reliable detections
- **Target Extrapolation Model**: Aggressive fallback for poor detection scenarios
- **Servo Controller**: Controls pan/tilt servos based on predicted coordinates

## Features

- **Frame Labeling System**: Prevents aiming at outdated frames
- **Dual Queue Architecture**: Separate queues for raw and processed frames
- **Adaptive Targeting**: Switches between prediction models based on success rate
- **Mock Testing Framework**: Complete testing system with simulated targets
- **Debug Mode**: Comprehensive debugging and performance monitoring

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Test Videos
```bash
python testing/mock_video_generator.py
```

### 3. Run Basic Test
```bash
# Test with camera
python main.py --debug

# Test with mock video
python main.py --debug --video data/mock_videos/jet_test_01.mp4

# List available test videos
python main.py --list-videos
```

### 4. Run Performance Tests
```bash
# Single performance test
python testing/test_runner.py --perf

# Full test suite
python testing/test_runner.py --all
```

## Usage Examples

### Testing with Mock Data
```bash
# Generate mock videos first
python testing/mock_video_generator.py

# Test specific video with debugging
python main.py --debug --video data/mock_videos/drone_test_01.mp4

# Run comprehensive test suite
python testing/test_runner.py --video-suite
```

### Real Hardware Integration
```bash
# Use real camera (disable mock YOLO when ready)
python main.py --no-mock

# Production mode (no debug output)
python main.py
```

## Configuration

Edit `config.json` to customize system parameters:

```json
{
    "detection": {
        "confidence_threshold": 0.5,
        "target_classes": ["aircraft", "drone", "helicopter"]
    },
    "tracking": {
        "success_rate_threshold": 0.8,
        "prediction_time_horizon": 0.1
    },
    "servo": {
        "pan_limits": [-90, 90],
        "tilt_limits": [-45, 45]
    }
}
```

## Debug Output

When running with `--debug`, you'll see:

```
=== NDS Sentry Debug Mode ===
Mock detection: True
Video file: data/mock_videos/jet_test_01.mp4
Threads started. Beginning control loop...

[2.3s] PREDICTION: Raw=(0.456,0.342) Final=(0.463,0.348) Conf=0.782 SuccessRate=0.734

=== DEBUG SUMMARY [10.0s] ===
Frames processed: 299 (29.9 FPS)
Detections: 254 (84.9%)
Predictions: 198
Extrapolations: 56
Current success rate: 0.756
```

## Testing Framework

The system includes comprehensive testing tools:

- **Mock Video Generator**: Creates realistic test videos with simulated aircraft
- **Mock YOLO Detector**: Simulates realistic detection behavior
- **Performance Testing**: Measures FPS, detection rates, and threading efficiency
- **Stress Testing**: Tests system under various load conditions

## Hardware Integration

To integrate with real hardware:

1. **Camera**: Update `src/image_capture.py` camera initialization
2. **YOLO Model**: Replace mock detector in `src/yolo_processor.py`
3. **Servo Control**: Implement actual servo communication in `src/servo_controller.py`

## Development

### Project Structure
```
├── main.py              # Main controller and entry point
├── src/                 # Core system modules
│   ├── image_capture.py # Camera/video capture thread
│   ├── yolo_processor.py# Target detection thread
│   ├── target_prediction.py  # High-accuracy prediction
│   ├── target_extrapolation.py  # Fallback extrapolation
│   ├── servo_controller.py    # Servo control
│   ├── mock_yolo.py     # Mock detector for testing
│   ├── queues.py        # Thread-safe queues
│   └── frame_data.py    # Data structures
├── testing/             # Testing framework
│   ├── mock_video_generator.py
│   └── test_runner.py
├── data/               # Test data directory
└── config.json         # System configuration
```

### Next Steps

1. Train and integrate actual YOLO model
2. Implement hardware-specific servo drivers
3. Add camera calibration routines
4. Optimize prediction algorithms based on real data
5. Add safety systems and emergency stops

## License

MIT License - See LICENSE file for details.

---

*This is a Purdue club project. Model binaries and training data remain private due to dataset privacy limitations.*