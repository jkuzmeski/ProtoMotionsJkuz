#!/usr/bin/env python3

"""
Simple test script to demonstrate the current skeleton retargeting output.
This will help us see exactly what output you're getting.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add ProtoMotions to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Import the retargeting function
    from data.scripts.retarget_treadmill_motion import create_skeleton_motion_from_positions
    
    print("=" * 70)
    print("SKELETON MOTION RETARGETING TEST")
    print("=" * 70)
    
    # Create some simple test data (anatomically correct positions)
    n_frames = 10
    joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe']
    
    # Create anatomically correct test positions
    # Pelvis at height 0.9, ankles at height 0.1 (below pelvis)
    joint_positions = np.zeros((n_frames, len(joint_names), 3))
    
    for frame in range(n_frames):
        # Pelvis (root) - at hip height
        joint_positions[frame, 0] = [0.0, 0.0, 0.9]  # Pelvis
        
        # Left leg
        joint_positions[frame, 1] = [-0.1, 0.0, 0.8]  # L_Hip (slightly below pelvis)
        joint_positions[frame, 2] = [-0.1, 0.0, 0.5]  # L_Knee (mid-height)
        joint_positions[frame, 3] = [-0.1, 0.0, 0.1]  # L_Ankle (near ground)
        joint_positions[frame, 4] = [-0.1, 0.1, 0.0]  # L_Toe (on ground)
        
        # Right leg
        joint_positions[frame, 5] = [0.1, 0.0, 0.8]   # R_Hip (slightly below pelvis)
        joint_positions[frame, 6] = [0.1, 0.0, 0.5]   # R_Knee (mid-height)
        joint_positions[frame, 7] = [0.1, 0.0, 0.1]   # R_Ankle (near ground)
        joint_positions[frame, 8] = [0.1, 0.1, 0.0]   # R_Toe (on ground)
    
    print("INPUT DATA (Anatomically Correct):")
    print(f"Shape: {joint_positions.shape}")
    print("\nFirst frame positions:")
    for i, name in enumerate(joint_names):
        pos = joint_positions[0, i]
        print(f"  {name}: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")
    
    print(f"\nExpected: Ankles (Z=0.1) should be BELOW Pelvis (Z=0.9)")
    
    print("\n" + "="*70)
    print("RUNNING RETARGETING...")
    print("="*70)
    
    # Run the retargeting
    sk_motion = create_skeleton_motion_from_positions(
        joint_positions, 
        joint_names, 
        fps=200, 
        render=False
    )
    
    print("\n" + "="*70)
    print("ANALYZING RESULTS...")
    print("="*70)
    
    # Get the result positions
    result_positions = sk_motion.global_translation[0].numpy()
    
    print("\nFinal skeleton positions (first frame):")
    for i, name in enumerate(sk_motion.skeleton_tree.node_names):
        pos = result_positions[i]
        print(f"  {name}: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")
    
    # Check anatomical correctness
    pelvis_idx = sk_motion.skeleton_tree.node_names.index('Pelvis')
    pelvis_z = result_positions[pelvis_idx, 2]
    
    print(f"\nANATOMICAL CORRECTNESS CHECK:")
    print(f"Pelvis height: {pelvis_z:.3f}")
    
    ankle_names = ['L_Ankle', 'R_Ankle']
    for ankle_name in ankle_names:
        if ankle_name in sk_motion.skeleton_tree.node_names:
            ankle_idx = sk_motion.skeleton_tree.node_names.index(ankle_name)
            ankle_z = result_positions[ankle_idx, 2]
            is_correct = ankle_z < pelvis_z
            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            print(f"{ankle_name}: {ankle_z:.3f} - {status}")
    
    # Find lowest and highest points
    z_values = result_positions[:, 2]
    lowest_idx = np.argmin(z_values)
    highest_idx = np.argmax(z_values)
    
    print(f"\nLowest point: {sk_motion.skeleton_tree.node_names[lowest_idx]} at Z={z_values[lowest_idx]:.3f}")
    print(f"Highest point: {sk_motion.skeleton_tree.node_names[highest_idx]} at Z={z_values[highest_idx]:.3f}")
    
    # Overall assessment
    feet_below_pelvis = all(
        result_positions[sk_motion.skeleton_tree.node_names.index(ankle), 2] < pelvis_z
        for ankle in ankle_names
        if ankle in sk_motion.skeleton_tree.node_names
    )
    
    print(f"\n" + "="*70)
    print(f"OVERALL RESULT: {'SUCCESS' if feet_below_pelvis else 'FAILURE'}")
    if feet_below_pelvis:
        print("✓ Ankles are correctly positioned below pelvis")
    else:
        print("✗ Anatomically incorrect - ankles above pelvis")
    print("="*70)
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
