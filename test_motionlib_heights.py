#!/usr/bin/env python3

"""
Test that directly loads the motion into MotionLib to check final heights.
This simulates exactly what happens in Isaac Lab.
"""

import sys
sys.path.append('d:/Isaac/IsaacLab2.1.2/source')

import numpy as np
import torch
from isaaclab.utils.motion import SkeletonMotion

def test_motionlib_heights():
    """Test the heights that MotionLib would see."""
    print("=== Testing MotionLib Height Processing ===\n")
    
    # Step 1: Load the motion file exactly like MotionLib does
    motion_file = "data/scripts/data2retarget/retargeted_motion.npy" 
    print(f"Loading motion: {motion_file}")
    
    try:
        # Load the motion exactly like MotionLib._load_motion_file does
        motion = SkeletonMotion.from_file(motion_file)
        print(f"✅ Motion loaded successfully")
        print(f"   Frames: {motion.num_frames}")
        print(f"   FPS: {motion.fps}")
        
        # Check the root translation heights BEFORE any processing
        print(f"\n📊 Root translation heights (before any processing):")
        for i in [0, 10, 50, 100]:
            if i < motion.num_frames:
                root_pos = motion.root_translation[i].numpy()
                print(f"   Frame {i:3d}: [{root_pos[0]:7.3f}, {root_pos[1]:7.3f}, {root_pos[2]:7.3f}]")
        
        # Step 2: Check what happens with fix_motion_heights
        print(f"\n🔧 Testing fix_motion_heights function...")
        
        # Get the minimum height (what fix_motion_heights uses)
        body_heights = motion.global_translation[..., 2]
        min_height = body_heights.min()
        print(f"   Minimum body height: {min_height:.4f}")
        
        # This is what fix_motion_heights would do
        adjusted_root_translation = motion.root_translation.clone()
        adjusted_root_translation[:, 2] -= min_height
        
        print(f"\n📊 Root heights AFTER fix_motion_heights adjustment:")
        for i in [0, 10, 50, 100]:
            if i < motion.num_frames:
                root_pos = adjusted_root_translation[i].numpy()
                print(f"   Frame {i:3d}: [{root_pos[0]:7.3f}, {root_pos[1]:7.3f}, {root_pos[2]:7.3f}]")
        
        # Step 3: Check if fix_motion_heights would be called
        print(f"\n⚙️  MotionLib behavior analysis:")
        print(f"   - fix_motion_heights default: False")
        print(f"   - If fix_heights=True: Root height would be ~{adjusted_root_translation[0, 2]:.3f}")
        print(f"   - If fix_heights=False: Root height would be ~{motion.root_translation[0, 2]:.3f}")
        
        # Check ankle heights vs pelvis
        print(f"\n🦵 Joint height analysis:")
        global_pos = motion.global_translation[0].numpy()
        joint_names = motion.skeleton_tree.node_names
        
        for i, name in enumerate(joint_names):
            height = global_pos[i, 2]
            print(f"   {name}: {height:.4f}m")
        
        # Find minimum joint height
        min_joint_height = global_pos[:, 2].min()
        max_joint_height = global_pos[:, 2].max()
        print(f"\n   Minimum joint height: {min_joint_height:.4f}m")
        print(f"   Maximum joint height: {max_joint_height:.4f}m")
        print(f"   Height range: {max_joint_height - min_joint_height:.4f}m")
        
        if min_joint_height < 0.1:
            print(f"   ⚠️  Some joints are very low (<0.1m) - fix_motion_heights would activate")
        else:
            print(f"   ✅ All joints have reasonable heights")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_motionlib_heights()
