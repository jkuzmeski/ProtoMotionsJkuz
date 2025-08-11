#!/usr/bin/env python3

"""
Test script to verify MuJoCo joint angle mapping logic.
"""

import numpy as np

def test_joint_mapping():
    """Test the joint mapping logic we implemented."""
    
    print("=== Testing MuJoCo Joint Angle Mapping ===")
    
    # Simulate MuJoCo qpos array structure
    # qpos[0:7] = root position (3) + quaternion (4) 
    # qpos[7:] = joint angles in groups of 3
    
    # Example qpos[7:] with 8 joints * 3 DOF = 24 values
    example_qpos_joints = np.array([
        # L_Hip (3 angles)
        0.1, 0.2, 0.3,
        # L_Knee (3 angles)  
        0.4, 0.5, 0.6,
        # L_Ankle (3 angles)
        0.7, 0.8, 0.9,
        # L_Toe (3 angles)
        0.11, 0.12, 0.13,
        # R_Hip (3 angles)
        0.14, 0.15, 0.16,
        # R_Knee (3 angles)
        0.17, 0.18, 0.19,
        # R_Ankle (3 angles)
        0.20, 0.21, 0.22,
        # R_Toe (3 angles)
        0.23, 0.24, 0.25
    ])
    
    print(f"Example qpos[7:] array shape: {example_qpos_joints.shape}")
    print(f"Example qpos[7:] values: {example_qpos_joints}")
    
    # Our mapping from MuJoCo qpos indices to skeleton joints
    mujoco_to_skeleton_mapping = {
        'L_Hip': (0, 1),    # qpos[7:10] -> skeleton joint 1 (L_Hip)
        'L_Knee': (3, 2),   # qpos[10:13] -> skeleton joint 2 (L_Knee)  
        'L_Ankle': (6, 3),  # qpos[13:16] -> skeleton joint 3 (L_Ankle)
        'L_Toe': (9, 4),    # qpos[16:19] -> skeleton joint 4 (L_Toe)
        'R_Hip': (12, 5),   # qpos[19:22] -> skeleton joint 5 (R_Hip)
        'R_Knee': (15, 6),  # qpos[22:25] -> skeleton joint 6 (R_Knee)
        'R_Ankle': (18, 7), # qpos[25:28] -> skeleton joint 7 (R_Ankle)
        'R_Toe': (21, 8),   # qpos[28:31] -> skeleton joint 8 (R_Toe)
    }
    
    print("\n=== Testing Joint Extraction ===")
    
    for joint_name, (mujoco_idx, skeleton_idx) in mujoco_to_skeleton_mapping.items():
        if mujoco_idx + 2 < len(example_qpos_joints):
            joint_angles = example_qpos_joints[mujoco_idx:mujoco_idx + 3]
            print(f"{joint_name:8}: MuJoCo idx {mujoco_idx:2}-{mujoco_idx+2:2} -> Skeleton {skeleton_idx} = {joint_angles}")
        else:
            print(f"{joint_name:8}: MuJoCo idx {mujoco_idx:2}-{mujoco_idx+2:2} -> OUT OF BOUNDS!")
    
    print("\n=== Expected MuJoCo Joint Order ===")
    print("Based on SMPL lower body model:")
    print("Joint 0: Pelvis (root) - handled separately from qpos[0:7]")
    print("Joint 1: L_Hip")  
    print("Joint 2: L_Knee")
    print("Joint 3: L_Ankle")
    print("Joint 4: L_Toe")
    print("Joint 5: R_Hip")
    print("Joint 6: R_Knee") 
    print("Joint 7: R_Ankle")
    print("Joint 8: R_Toe")
    
    print("\n=== Verification ===")
    print("If our mapping is correct, we should extract:")
    print("L_Hip angles:   [0.1, 0.2, 0.3]")
    print("L_Knee angles:  [0.4, 0.5, 0.6]") 
    print("L_Ankle angles: [0.7, 0.8, 0.9]")
    print("L_Toe angles:   [0.11, 0.12, 0.13]")
    print("R_Hip angles:   [0.14, 0.15, 0.16]")
    print("R_Knee angles:  [0.17, 0.18, 0.19]")
    print("R_Ankle angles: [0.20, 0.21, 0.22]")
    print("R_Toe angles:   [0.23, 0.24, 0.25]")

if __name__ == "__main__":
    test_joint_mapping()
