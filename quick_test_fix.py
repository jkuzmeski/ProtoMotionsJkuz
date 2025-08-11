#!/usr/bin/env python3

# Quick test of the fix
import numpy as np
import sys
import os

# Set the path to import our module
sys.path.insert(0, os.path.dirname(__file__))

def quick_test():
    """
    Quick test to see if our fix works
    """
    try:
        # Load test data
        joint_positions = np.load("data/scripts/data2retarget/overground_motion.npy")[:5]  # Just 5 frames
        joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe']
        
        print("[QUICK_TEST] Testing with 5 frames...")
        print(f"Input shape: {joint_positions.shape}")
        
        # Import and test our function
        from data.scripts.retarget_treadmill_motion import create_skeleton_motion_from_positions
        
        # Run the function
        sk_motion = create_skeleton_motion_from_positions(
            joint_positions, 
            joint_names, 
            fps=200, 
            render=False
        )
        
        print(f"[SUCCESS] Created motion with {len(sk_motion.root_translation)} frames")
        
        # Check the positioning
        frame_0_global = sk_motion.global_translation[0].numpy()
        pelvis_pos = frame_0_global[0]
        l_ankle_pos = frame_0_global[3]  # L_Ankle should be index 3
        r_ankle_pos = frame_0_global[7]  # R_Ankle should be index 7
        
        print(f"\n[RESULT] Frame 0 positions:")
        print(f"  Pelvis: [{pelvis_pos[0]:.4f}, {pelvis_pos[1]:.4f}, {pelvis_pos[2]:.4f}]")
        print(f"  L_Ankle: [{l_ankle_pos[0]:.4f}, {l_ankle_pos[1]:.4f}, {l_ankle_pos[2]:.4f}]")
        print(f"  R_Ankle: [{r_ankle_pos[0]:.4f}, {r_ankle_pos[1]:.4f}, {r_ankle_pos[2]:.4f}]")
        
        # Check if fix worked
        if l_ankle_pos[2] < pelvis_pos[2] and r_ankle_pos[2] < pelvis_pos[2]:
            print(f"\n[SUCCESS] Fix worked! Ankles are below pelvis.")
            return True
        else:
            print(f"\n[ERROR] Fix failed. Ankles still above pelvis.")
            return False
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    print(f"\n[FINAL] Test {'PASSED' if success else 'FAILED'}")
