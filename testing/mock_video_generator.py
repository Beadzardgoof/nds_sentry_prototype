#!/usr/bin/env python3

import cv2
import numpy as np
import os
from typing import Tuple, List
import random
import math

class MockVideoGenerator:
    """Generates mock test videos with simulated aircraft/drone targets for testing"""
    
    def __init__(self, output_dir: str = "../data/mock_videos"):
        self.output_dir = output_dir
        self.width = 1280
        self.height = 720
        self.fps = 30
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_jet_video(self, filename: str = "jet_test_01.mp4", duration: int = 60):
        """Generate video with fast-moving jet aircraft"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(self.output_dir, filename), fourcc, self.fps, (self.width, self.height))
        
        total_frames = duration * self.fps
        
        # Jet parameters - fast linear motion
        start_x = -100
        end_x = self.width + 100
        y = self.height // 2 + random.randint(-100, 100)
        
        for frame_num in range(total_frames):
            # Create sky background
            frame = self.create_sky_background()
            
            # Calculate jet position (linear motion)
            progress = frame_num / total_frames
            jet_x = int(start_x + (end_x - start_x) * progress)
            jet_y = int(y + 20 * math.sin(progress * math.pi * 4))  # Slight vertical oscillation
            
            # Draw jet (simple representation)
            self.draw_jet(frame, jet_x, jet_y)
            
            out.write(frame)
            
        out.release()
        print(f"Generated mock jet video: {filename}")
        
    def generate_drone_video(self, filename: str = "drone_test_01.mp4", duration: int = 45):
        """Generate video with slower-moving drone target"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(self.output_dir, filename), fourcc, self.fps, (self.width, self.height))
        
        total_frames = duration * self.fps
        
        # Drone parameters - circular/hovering motion
        center_x = self.width // 2
        center_y = self.height // 2
        radius = 200
        
        for frame_num in range(total_frames):
            # Create sky background
            frame = self.create_sky_background()
            
            # Calculate drone position (circular motion)
            angle = (frame_num / total_frames) * 2 * math.pi * 3  # 3 full circles
            drone_x = int(center_x + radius * math.cos(angle))
            drone_y = int(center_y + radius * math.sin(angle))
            
            # Add some random drift
            drone_x += random.randint(-10, 10)
            drone_y += random.randint(-10, 10)
            
            # Draw drone (simple representation)
            self.draw_drone(frame, drone_x, drone_y)
            
            out.write(frame)
            
        out.release()
        print(f"Generated mock drone video: {filename}")
        
    def generate_mixed_targets_video(self, filename: str = "mixed_targets_01.mp4", duration: int = 90):
        """Generate video with multiple different target types"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(self.output_dir, filename), fourcc, self.fps, (self.width, self.height))
        
        total_frames = duration * self.fps
        
        for frame_num in range(total_frames):
            # Create sky background
            frame = self.create_sky_background()
            
            # Simulate different scenarios based on time
            time_progress = frame_num / total_frames
            
            if time_progress < 0.3:
                # First 30% - drone hovering
                angle = time_progress * 2 * math.pi * 10
                drone_x = int(self.width // 2 + 150 * math.cos(angle))
                drone_y = int(self.height // 2 + 100 * math.sin(angle))
                self.draw_drone(frame, drone_x, drone_y)
                
            elif time_progress < 0.6:
                # Middle 30% - jet passing through
                jet_progress = (time_progress - 0.3) / 0.3
                jet_x = int(-100 + (self.width + 200) * jet_progress)
                jet_y = int(self.height // 3)
                self.draw_jet(frame, jet_x, jet_y)
                
            else:
                # Final 40% - helicopter
                heli_progress = (time_progress - 0.6) / 0.4
                heli_x = int(self.width // 4 + (self.width // 2) * heli_progress)
                heli_y = int(self.height // 2 + 50 * math.sin(heli_progress * math.pi * 8))
                self.draw_helicopter(frame, heli_x, heli_y)
                
            out.write(frame)
            
        out.release()
        print(f"Generated mock mixed targets video: {filename}")
        
    def create_sky_background(self) -> np.ndarray:
        """Create a realistic sky background with some clouds"""
        # Create gradient sky
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Blue sky gradient
        for y in range(self.height):
            intensity = int(255 - (y / self.height) * 100)  # Darker at bottom
            frame[y, :] = [intensity, intensity - 20, 200]  # Sky blue
            
        # Add some random cloud-like noise
        noise = np.random.randint(0, 30, (self.height, self.width, 3))
        frame = cv2.add(frame, noise.astype(np.uint8))
        
        return frame
        
    def draw_jet(self, frame: np.ndarray, x: int, y: int):
        """Draw a simple jet aircraft representation"""
        if 0 <= x < self.width and 0 <= y < self.height:
            # Main fuselage
            cv2.rectangle(frame, (x-15, y-3), (x+15, y+3), (128, 128, 128), -1)
            # Wings
            cv2.rectangle(frame, (x-5, y-12), (x+5, y+12), (100, 100, 100), -1)
            # Nose
            cv2.circle(frame, (x+15, y), 3, (80, 80, 80), -1)
            # Exhaust trail (simple line)
            cv2.line(frame, (x-15, y), (x-30, y), (200, 200, 255), 2)
            
    def draw_drone(self, frame: np.ndarray, x: int, y: int):
        """Draw a simple drone representation"""
        if 0 <= x < self.width and 0 <= y < self.height:
            # Main body
            cv2.circle(frame, (x, y), 8, (60, 60, 60), -1)
            # Four rotors
            rotor_positions = [(x-12, y-12), (x+12, y-12), (x-12, y+12), (x+12, y+12)]
            for rx, ry in rotor_positions:
                cv2.circle(frame, (rx, ry), 4, (40, 40, 40), -1)
                # Rotor blur (spinning effect)
                cv2.circle(frame, (rx, ry), 8, (200, 200, 200), 1)
                
    def draw_helicopter(self, frame: np.ndarray, x: int, y: int):
        """Draw a simple helicopter representation"""
        if 0 <= x < self.width and 0 <= y < self.height:
            # Main body
            cv2.ellipse(frame, (x, y), (20, 8), 0, 0, 360, (100, 100, 100), -1)
            # Main rotor (top)
            cv2.line(frame, (x-25, y-15), (x+25, y-15), (200, 200, 200), 2)
            cv2.circle(frame, (x, y-15), 3, (80, 80, 80), -1)
            # Tail rotor
            cv2.circle(frame, (x+25, y), 5, (200, 200, 200), 1)
            # Tail boom
            cv2.line(frame, (x+10, y), (x+25, y), (80, 80, 80), 3)

def main():
    """Generate all mock test videos"""
    generator = MockVideoGenerator()
    
    print("Generating mock test videos...")
    generator.generate_jet_video()
    generator.generate_drone_video()
    generator.generate_mixed_targets_video()
    
    print("Mock video generation complete!")
    print("Videos saved to: data/mock_videos/")

if __name__ == "__main__":
    main()