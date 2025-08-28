# Claude Code Project Guide

## Overview
This is the NDS Sentry Prototype - a modular target tracking system for a two-servo sentry system with real-time target detection, prediction, and servo control.

## Project Setup

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

### Key Dependencies
- opencv-python>=4.5.0
- numpy>=1.21.0
- ultralytics>=8.0.0 (YOLO)
- torch>=1.9.0
- torchvision>=0.10.0

## Common Commands

### Development Commands
```bash
# Run basic test with debug output
python main.py --debug

# Test with mock video
python main.py --debug --video data/mock_videos/jet_test_01.mp4

# List available test videos
python main.py --list-videos

# Generate test videos
python testing/mock_video_generator.py

# Run performance tests
python testing/test_runner.py --perf

# Run full test suite
python testing/test_runner.py --all
```

### Testing Commands
```bash
# Single performance test
python testing/test_runner.py --perf

# Test all mock videos
python testing/test_runner.py --video-suite

# Comprehensive testing
python testing/test_runner.py --all

# Custom duration test
python testing/test_runner.py --perf --duration 60
```

## Project Structure
```
├── main.py                    # Main entry point
├── src/                      # Core system modules
│   ├── image_capture.py      # Camera/video capture
│   ├── yolo_processor.py     # Target detection
│   ├── target_prediction.py  # High-accuracy prediction
│   ├── target_extrapolation.py # Fallback extrapolation
│   ├── servo_controller.py   # Servo control
│   ├── mock_yolo.py         # Mock detector for testing
│   ├── queues.py            # Thread-safe queues
│   └── frame_data.py        # Data structures
├── testing/                  # Testing framework
│   ├── mock_video_generator.py
│   └── test_runner.py
├── data/                    # Test data directory
└── config.json             # System configuration
```

## Architecture Notes
- Uses 20-thread dynamic pool with automatic allocation
- Multi-stage freshness validation system
- Predictive mode switching based on failure rate analysis
- Mock testing framework with realistic target simulation

## Development Workflow
1. Generate test videos first: `python testing/mock_video_generator.py`
2. Run with debug mode for development: `python main.py --debug`
3. Use specific test videos: `python main.py --video data/mock_videos/jet_test_01.mp4 --debug`
4. Run performance tests: `python testing/test_runner.py --perf`

## Hardware Integration
To integrate with real hardware, update:
- Camera initialization in `src/image_capture.py`
- YOLO model in `src/yolo_processor.py` (replace mock)
- Servo communication in `src/servo_controller.py`

## Debugging
Debug mode shows:
- Frame processing rates and FPS
- Detection success rates
- Threading performance
- Real-time prediction accuracy

Use `--debug` flag for comprehensive output during development.