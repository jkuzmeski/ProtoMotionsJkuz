#!/usr/bin/env python3

import numpy as np
import torch
from pathlib import Path
from data.scripts.convert_to_isaac import create_motion_from_txt

def test_ik_conversion():
    """Test the IK-based conversion step by step"""
    
    print("=== TESTING IK CONVERSION ===")
    
    try:
        print("1. Loading motion data...")
        motion = create_motion_from_txt(
            'data/scripts/data2retarget/S02_3-0ms_positons_lowerbody.txt', 
            'data/scripts/data2retarget/S02_tpose_positons_lowerbody.txt', 
            3.0
        )
        print(f"   Motion loaded successfully!")
        print(f"   Frames: {motion.global_translation.shape[0]}")
        print(f"   Joints: {motion.global_translation.shape[1]}")
        print(f"   Rotations shape: {motion.global_rotation.shape}")
        
        # Check if we have non-identity rotations
        identity_check = motion.global_rotation[:, :, 0]  # w component
        non_identity_count = torch.sum(torch.abs(identity_check - 1.0) > 1e-6)
        print(f"   Non-identity rotations: {non_identity_count}")
        
        print("2. Attempting retargeting...")
        from data.scripts.convert_to_isaac import retarget_motion_to_robot
        
        retargeted_motion = retarget_motion_to_robot(motion, 'smpl_humanoid_lower_body', render=False)
        print(f"   Retargeting completed successfully!")
        print(f"   Retargeted frames: {retargeted_motion.global_translation.shape[0]}")
        print(f"   Retargeted joints: {retargeted_motion.global_translation.shape[1]}")
        
        # Save a simple debug file
        retargeted_positions = retargeted_motion.global_translation.numpy()
        np.savez(
            'debug_ik_test.npz',
            body_positions=retargeted_positions,
            body_names=retargeted_motion.skeleton_tree.node_names,
            fps=retargeted_motion.fps
        )
        print(f"   Debug file saved: debug_ik_test.npz")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ik_conversion()
    print(f"Test {'PASSED' if success else 'FAILED'}")
