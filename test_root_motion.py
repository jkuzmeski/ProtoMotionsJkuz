#!/usr/bin/env python3

"""
Test to verify root motion is working correctly.
"""

import numpy as np
import torch
from poselib.skeleton.skeleton3d import SkeletonMotion

def test_root_motion():
    """Test if the retargeted motion has proper root movement."""
    
    print("=== Testing Root Motion ===")
    
    try:
        # Load the latest retargeted motion
        motion = np.load('../motions/retargeted_motion_fixed_root.npy', allow_pickle=True).item()
        
        if 'rotation' in motion:
            rotations = motion['rotation']['arr']
            root_trans = motion['root_translation']['arr']
            
            print(f"Motion data shapes:")
            print(f"  Rotations: {rotations.shape}")
            print(f"  Root translations: {root_trans.shape}")
            
            print(f"\nRoot position progression (first 10 frames):")
            for i in range(min(10, len(root_trans))):
                pos = root_trans[i]
                print(f"  Frame {i:2}: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
            
            # Check if root is moving
            first_pos = root_trans[0]
            last_pos = root_trans[-1]
            distance_moved = np.linalg.norm(last_pos - first_pos)
            
            print(f"\nRoot movement analysis:")
            print(f"  First position: [{first_pos[0]:.4f}, {first_pos[1]:.4f}, {first_pos[2]:.4f}]")
            print(f"  Last position:  [{last_pos[0]:.4f}, {last_pos[1]:.4f}, {last_pos[2]:.4f}]")
            print(f"  Total distance moved: {distance_moved:.4f}m")
            
            if distance_moved > 0.01:  # More than 1cm movement
                print("  ✅ SUCCESS: Root is moving!")
            else:
                print("  ❌ PROBLEM: Root is not moving enough")
                
            # Check height range
            z_values = root_trans[:, 2]
            min_z = np.min(z_values)
            max_z = np.max(z_values)
            mean_z = np.mean(z_values)
            
            print(f"\nHeight analysis:")
            print(f"  Min height: {min_z:.4f}m")
            print(f"  Max height: {max_z:.4f}m")
            print(f"  Mean height: {mean_z:.4f}m")
            print(f"  Height variation: {max_z - min_z:.4f}m")
            
            # Expected human pelvis height should be around 0.8-1.2m
            if 0.7 < mean_z < 1.5:
                print("  ✅ Height looks reasonable for human pelvis")
            else:
                print("  ⚠️  Height seems unusual for human pelvis")
                
        else:
            print("❌ No rotation data found in motion file")
            
    except FileNotFoundError:
        print("❌ Motion file not found - retargeting may not have completed yet")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_root_motion()
