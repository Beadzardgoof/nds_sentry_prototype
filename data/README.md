# Mock Data Directory

This directory contains mock data for testing the NDS Sentry target tracking system.

## Directory Structure

- `mock_videos/` - Test videos for pipeline testing (jets, drones, aircraft)
- `test_datasets/` - Mock datasets for model training/validation
- `mock_targets/` - Individual target images for testing

## Mock Video Sources

Place test videos here with the following naming convention:
- `jet_test_01.mp4` - Fast-moving jet aircraft
- `drone_test_01.mp4` - Small drone targets  
- `helicopter_test_01.mp4` - Rotorcraft targets
- `mixed_targets_01.mp4` - Multiple target types

## Video Format Requirements

- Format: MP4, AVI, or MOV
- Resolution: 1280x720 minimum
- Frame rate: 30 FPS preferred
- Duration: 30-120 seconds for testing

## Usage

Videos are automatically detected by the testing framework and can be selected via configuration files.