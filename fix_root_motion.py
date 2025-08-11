#!/usr/bin/env python3

"""
Quick fix to update the existing motion file with proper root translations.
"""

import numpy as np
import torch
from poselib.skeleton.skeleton3d import SkeletonMotion

def fix_existing_motion():
    """Fix the existing motion by adding proper root translations."""
    
    print("=== Fixing Existing Motion Root Translations ===")
    
    try:
        # Load the current motion
        motion_data = np.load('data/motions/retargeted_with_rotations.npy', allow_pickle=True).item()
        
        print("Current motion loaded successfully")
        
        # Get the global translations (these are the correct positions)
        if 'global_translation' in motion_data:
            global_trans = motion_data['global_translation']['arr']
            print(f"Global translation shape: {global_trans.shape}")
            
            # Extract pelvis positions (index 0) as root translations
            pelvis_positions = global_trans[:, 0, :]  # Shape: (n_frames, 3)
            
            print(f"Extracted pelvis positions shape: {pelvis_positions.shape}")
            print("First 5 pelvis positions:")
            for i in range(5):
                pos = pelvis_positions[i]
                print(f"  Frame {i}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            
            # Update the root translation
            motion_data['root_translation']['arr'] = pelvis_positions
            
            # Save the fixed motion
            np.save('data/motions/retargeted_with_rotations_fixed.npy', motion_data, allow_pickle=True)
            
            print("\n✅ Fixed motion saved to: data/motions/retargeted_with_rotations_fixed.npy")
            
            # Verify the fix
            fixed_motion = np.load('data/motions/retargeted_with_rotations_fixed.npy', allow_pickle=True).item()
            fixed_root_trans = fixed_motion['root_translation']['arr']
            
            print("\nVerification - Fixed root translations:")
            for i in range(5):
                pos = fixed_root_trans[i]
                print(f"  Frame {i}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            
            # Check movement
            first_pos = fixed_root_trans[0]
            last_pos = fixed_root_trans[-1]
            distance_moved = np.linalg.norm(last_pos - first_pos)
            
            print(f"\nMovement analysis:")
            print(f"  Distance moved: {distance_moved:.4f}m")
            print(f"  Height range: {np.min(fixed_root_trans[:, 2]):.4f} to {np.max(fixed_root_trans[:, 2]):.4f}")
            
            if distance_moved > 0.01:
                print("  ✅ SUCCESS: Root is now moving properly!")
            else:
                print("  ⚠️  Root movement is still minimal")
                
        else:
            print("❌ No global_translation found in motion data")
            print(f"Available keys: {motion_data.keys()}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_existing_motion()
