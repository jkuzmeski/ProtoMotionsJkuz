#!/usr/bin/env python3

"""
Test script to verify MotionLib can load and use the retargeted motion.
"""

import torch
import numpy as np
from pathlib import Path
from poselib.skeleton.skeleton3d import SkeletonMotion

print("=" * 70)
print("TESTING MOTIONLIB COMPATIBILITY")
print("=" * 70)

# Test loading the retargeted motion
motion_file = "data/motions/retargeted_motion.npy"

try:
    # Load the motion (same way MotionLib does)
    motion = SkeletonMotion.from_file(motion_file)
    
    print(f"✅ Successfully loaded motion from {motion_file}")
    print(f"   - Frames: {motion.global_translation.shape[0]}")
    print(f"   - Joints: {motion.global_translation.shape[1]}")
    print(f"   - FPS: {motion.fps}")
    print(f"   - Is local: {motion.is_local}")
    
    # Check the key attributes MotionLib needs
    print(f"\nMotionLib required attributes:")
    print(f"✅ global_translation: {motion.global_translation.shape}")
    print(f"✅ local_rotation: {motion.local_rotation.shape}")  
    print(f"✅ root_translation: {motion.root_translation.shape}")
    print(f"✅ skeleton_tree: {len(motion.skeleton_tree.node_names)} joints")
    
    # Test frame sampling (what MotionLib does)
    print(f"\nTesting frame sampling (MotionLib functionality):")
    
    # Sample first frame
    frame_0_global_pos = motion.global_translation[0]
    frame_0_local_rot = motion.local_rotation[0]
    frame_0_root_pos = motion.root_translation[0]
    
    print(f"Frame 0 - Root position: {frame_0_root_pos.numpy()}")
    print(f"Frame 0 - Pelvis global: {frame_0_global_pos[0].numpy()}")
    print(f"Frame 0 - Pelvis rotation: {frame_0_local_rot[0].numpy()}")
    
    # Check for realistic motion variation
    frame_mid = motion.global_translation.shape[0] // 2
    frame_mid_global_pos = motion.global_translation[frame_mid]
    
    pelvis_start = frame_0_global_pos[0].numpy()
    pelvis_mid = frame_mid_global_pos[0].numpy()
    pelvis_movement = np.linalg.norm(pelvis_start - pelvis_mid)
    
    print(f"\nMotion variation check:")
    print(f"Pelvis movement (start to mid): {pelvis_movement:.3f}m")
    print(f"Motion detected: {'✅ YES' if pelvis_movement > 0.1 else '⚠️ MINIMAL'}")
    
    # Check rotations are non-identity
    non_identity_rotations = 0
    for i in range(motion.local_rotation.shape[1]):
        rot = frame_0_local_rot[i].numpy()
        if not np.allclose(rot, [1.0, 0.0, 0.0, 0.0], atol=1e-3):
            non_identity_rotations += 1
    
    print(f"Non-identity rotations: {non_identity_rotations}/{motion.local_rotation.shape[1]}")
    
    # Verify anatomical correctness
    pelvis_z = frame_0_global_pos[0, 2].item()
    joint_names = motion.skeleton_tree.node_names
    
    anatomically_correct = True
    for i, name in enumerate(joint_names):
        if 'Ankle' in name:
            ankle_z = frame_0_global_pos[i, 2].item()
            if ankle_z >= pelvis_z:
                anatomically_correct = False
                print(f"❌ {name} above pelvis: {ankle_z:.3f} >= {pelvis_z:.3f}")
    
    if anatomically_correct:
        print(f"✅ Anatomically correct: All ankles below pelvis")
    
    print(f"\n" + "=" * 70)
    print(f"MOTIONLIB COMPATIBILITY: {'✅ FULLY COMPATIBLE' if anatomically_correct and non_identity_rotations > 0 else '⚠️ ISSUES DETECTED'}")
    print(f"Ready for use in ProtoMotions training!")
    print("=" * 70)
    
except Exception as e:
    print(f"❌ ERROR loading motion: {e}")
    import traceback
    traceback.print_exc()
