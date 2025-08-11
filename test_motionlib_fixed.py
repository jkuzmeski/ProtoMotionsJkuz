#!/usr/bin/env python3

"""Test the fixed motion in MotionLib to verify root motion works."""

import numpy as np
import torch
import sys
import os

# Add Isaac Lab to path
sys.path.append('d:/Isaac/IsaacLab2.1.2/source')

# Import MotionLib components
from isaaclab.utils.motion import SkeletonMotion

def test_fixed_motion():
    """Test the retargeted motion with fixed root translation."""
    print("=== Testing Fixed Root Motion in MotionLib ===\n")
    
    # Load the retargeted motion
    motion_file = "data/scripts/data2retarget/retargeted_motion.npy"
    
    if not os.path.exists(motion_file):
        print(f"ERROR: Motion file not found: {motion_file}")
        return
    
    print(f"Loading motion from: {motion_file}")
    motion_data = np.load(motion_file, allow_pickle=True).item()
    
    # Create SkeletonMotion from the data
    print("Creating SkeletonMotion...")
    
    try:
        # Convert the nested dictionary structure to expected format
        root_translation = motion_data['root_translation']['arr'] if isinstance(motion_data['root_translation'], dict) else motion_data['root_translation']
        rotation = motion_data['rotation']['arr'] if isinstance(motion_data['rotation'], dict) else motion_data['rotation']
        
        print(f"Root translation shape: {root_translation.shape}")
        print(f"Rotation shape: {rotation.shape}")
        
        # Convert to tensors
        root_translation = torch.from_numpy(root_translation).float()
        rotation = torch.from_numpy(rotation).float()
        
        # Create SkeletonMotion
        motion = SkeletonMotion.from_dict({
            'root_translation': root_translation,
            'rotation': rotation,
            'global_velocity': motion_data.get('global_velocity', {}),
            'global_angular_velocity': motion_data.get('global_angular_velocity', {}),
            'skeleton_tree': motion_data['skeleton_tree'],
            'is_local': motion_data.get('is_local', True),
            'fps': motion_data.get('fps', 30),
        })
        
        print(f"✅ SkeletonMotion created successfully!")
        print(f"   Frames: {motion.num_frames}")
        print(f"   Duration: {motion.num_frames / motion.fps:.2f} seconds")
        print(f"   FPS: {motion.fps}")
        
        # Test root position at different frames
        print("\n=== Root Position Analysis ===")
        test_frames = [0, 10, 50, 100, motion.num_frames-1]
        
        for frame in test_frames:
            if frame < motion.num_frames:
                root_pos = motion.root_translation[frame]
                print(f"Frame {frame:3d}: [{root_pos[0]:7.3f}, {root_pos[1]:7.3f}, {root_pos[2]:7.3f}]")
        
        # Calculate movement statistics
        root_positions = motion.root_translation
        x_movement = root_positions[-1, 0] - root_positions[0, 0]
        y_movement = root_positions[-1, 1] - root_positions[0, 1]
        z_movement = root_positions[-1, 2] - root_positions[0, 2]
        
        print(f"\n=== Movement Summary ===")
        print(f"Total X movement: {x_movement:.3f} meters")
        print(f"Total Y movement: {y_movement:.3f} meters")
        print(f"Total Z movement: {z_movement:.3f} meters")
        print(f"Total distance: {np.sqrt(x_movement**2 + y_movement**2):.3f} meters")
        
        # Verify motion is not static
        if abs(x_movement) > 0.1 or abs(y_movement) > 0.1:
            print("\n✅ SUCCESS: Root motion is working! The character moves across the scene.")
        else:
            print("\n❌ WARNING: Root motion appears limited. Character may still be stuck.")
            
        return True
        
    except Exception as e:
        print(f"❌ ERROR creating SkeletonMotion: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_fixed_motion()
