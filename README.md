# NDS Sentry Prototype

A modular target tracking system for a two-servo sentry system. This prototype implements a complete threading pipeline for real-time target detection, prediction, and servo control.

## Architecture

### Threading Pipeline
```
Image Capture → Frame Queue → YOLO Processors (Dynamic: 12-15) → Processed Frame Queue
     ↓              ↓                    ↓                              ↓
  (frame,        stores           (target_coords,                stores target
frame_label)   raw frames      processed_frame_label,            data with
                [freshness]      confidence)                    timestamps
                                    [freshness]                 [freshness]
                                         ↓                          ↓
                              Prediction/Extrapolation        Master Controller
                              Thread Pool (1-6)               (Servo Control)
                                         ↓                          ↓
                              failure_rate > 0.8?           Interprets:
                              ├─ NO → Coordinates           - Target coordinates
                              └─ YES → Scan patterns        - Scan patterns
                                                             - Pattern types
```

**Dynamic Thread Pool (20 total):**
- Image Capture: 1 thread
- YOLO Processors: 12-15 threads (dynamic allocation)
- Prediction: 1-4 threads (lightweight)
- Extrapolation: 1-2 threads (scan pattern generation)
- Master Controller: 1 thread (servo control)
- Standby Pool: 2 threads (hot-swap capability)

### Key Components

- **Dynamic Thread Pool Manager**: Manages 20 threads with automatic allocation and hot-swapping
- **Master Controller**: Pure servo control with scan pattern interpretation
- **Image Capture Thread**: Captures frames from camera/video with temporal labeling
- **YOLO Processor Pool**: Multiple threads for parallel target detection (maximized allocation)
- **Prediction Threads**: Lightweight coordinate prediction for stable tracking (1-4 threads)
- **Extrapolation Threads**: Scan pattern generation for search operations (1-2 threads)
- **Freshness Validation**: Three-stage frame staleness prevention system
- **Failure Rate Prediction**: Proactive mode switching based on trend analysis

## Features

- **Dynamic Thread Management**: 20-thread pool with automatic resource reallocation
- **Multi-Stage Freshness Validation**: Three-point staleness prevention system
- **Predictive Mode Switching**: Trend-based failure rate analysis (0.8 threshold)
- **Scan Pattern System**: Extrapolation threads generate search patterns for target recovery
- **Hot-Swap Threading**: 2 standby threads enable instant mode transitions
- **Maximized YOLO Processing**: Dynamic allocation prioritizes detection throughput
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

The system includes comprehensive testing tools for validation and performance analysis:

### Testing Scripts

#### Main Application (`main.py`)
Run the core sentry system with various options:

```bash
# Basic usage
python main.py [OPTIONS]

# Available flags:
--video, -v PATH      # Use specific video file for testing
--debug, -d          # Enable detailed debug output and statistics
--no-mock            # Use real YOLO model instead of mock detector
--list-videos, -l    # List all available test videos
```

**Examples:**
```bash
# Test with camera and debug output
python main.py --debug

# Test with specific video file
python main.py --video data/mock_videos/jet_test_01.mp4 --debug

# Production mode with real YOLO
python main.py --no-mock

# List available test videos
python main.py --list-videos
```

#### Test Runner (`testing/test_runner.py`)
Automated testing suite for performance analysis:

```bash
# Basic usage
python testing/test_runner.py [OPTIONS]

# Available flags:
--perf               # Run single performance test
--video-suite        # Test all available mock videos
--stress             # Run threading stress test
--all                # Run comprehensive test suite
--video, -v PATH     # Test specific video file
--duration, -d SECS  # Set test duration (default: 30 seconds)
```

**Examples:**
```bash
# Quick performance test
python testing/test_runner.py --perf

# Test all mock videos
python testing/test_runner.py --video-suite

# Comprehensive testing (all tests)
python testing/test_runner.py --all

# Custom duration test
python testing/test_runner.py --perf --duration 60
```

### Test Data Generation

#### Mock Video Generator (`testing/mock_video_generator.py`)
Creates realistic test videos with simulated aircraft targets:

**Generated Video Types:**
- `jet_test_01.mp4` - Fast-moving jet with linear trajectory (60s)
- `drone_test_01.mp4` - Hovering drone with circular motion (45s)
- `mixed_targets_01.mp4` - Multiple target types in sequence (90s)

**Usage:**
```bash
# Generate all test videos
python testing/mock_video_generator.py

# Videos are saved to: data/mock_videos/
```

**Video Features:**
- Realistic sky backgrounds with cloud noise
- Multiple aircraft types (jets, drones, helicopters)
- Various motion patterns (linear, circular, hovering)
- Configurable duration and target behavior
- HD resolution (1280x720) at 30 FPS

### Testing Capabilities

- **Mock YOLO Detector**: Simulates realistic detection behavior with confidence scoring
- **Performance Testing**: Measures FPS, detection rates, and threading efficiency
- **Stress Testing**: Tests system under various load conditions and queue sizes
- **Video Test Suite**: Automated testing across multiple video scenarios
- **Debug Mode**: Comprehensive real-time statistics and frame-by-frame analysis

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