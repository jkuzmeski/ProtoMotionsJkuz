#!/usr/bin/env python3

import sys
import os

# Add the current directory to path to import the retargeting script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from pathlib import Path

# Import specific functions from our retargeting script
try:
    from data.scripts.retarget_treadmill_motion import create_skeleton_motion_from_positions
    print("[TEST] Successfully imported retargeting functions")
except Exception as e:
    print(f"[ERROR] Failed to import: {e}")
    sys.exit(1)

def test_retargeting():
    """Test the retargeting with a small dataset"""
    
    # Load the motion data
    input_file = "data/scripts/data2retarget/overground_motion.npy"
    if not Path(input_file).exists():
        print(f"[ERROR] Input file not found: {input_file}")
        return
        
    print(f"[TEST] Loading test data from {input_file}")
    joint_positions = np.load(input_file)
    
    # Take only first 10 frames for quick testing
    joint_positions_test = joint_positions[:10]
    
    joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe']
    
    print(f"[TEST] Input shape: {joint_positions_test.shape}")
    print(f"[TEST] Joint names: {joint_names}")
    
    try:
        # Test the retargeting function
        print("[TEST] Starting retargeting...")
        sk_motion = create_skeleton_motion_from_positions(
            joint_positions_test, 
            joint_names, 
            fps=200, 
            render=False
        )
        
        # Check the results
        print(f"[TEST] Successfully created SkeletonMotion")
        print(f"[TEST] Motion frames: {len(sk_motion.root_translation)}")
        print(f"[TEST] Motion joints: {len(sk_motion.skeleton_tree.node_names)}")
        
        # Check frame 0 positioning
        global_pos_frame0 = sk_motion.global_translation[0].numpy()
        pelvis_pos = global_pos_frame0[0]  # Pelvis is index 0
        ankle_positions = []
        
        for i, joint_name in enumerate(sk_motion.skeleton_tree.node_names):
            pos = global_pos_frame0[i]
            print(f"[TEST]   {joint_name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            
            if 'Ankle' in joint_name:
                ankle_positions.append((joint_name, pos[2]))  # Z position
        
        # Check if feet are below pelvis (anatomically correct)
        pelvis_z = pelvis_pos[2]
        print(f"\n[TEST] Pelvis Z position: {pelvis_z:.4f}")
        
        for joint_name, ankle_z in ankle_positions:
            if ankle_z > pelvis_z:
                print(f"[ERROR] {joint_name} is above pelvis! (Z={ankle_z:.4f} > {pelvis_z:.4f})")
                return False
            else:
                print(f"[SUCCESS] {joint_name} is below pelvis (Z={ankle_z:.4f} < {pelvis_z:.4f})")
        
        print(f"\n[SUCCESS] Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Retargeting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_retargeting()
    if not success:
        sys.exit(1)
