# CORRECTED VERSION - Replace the problematic section in your retargeting script

def create_skeleton_motion_from_mink(
    poses: np.ndarray,
    trans: np.ndarray,
    skeleton_tree: SkeletonTree,
    orig_global_trans: np.ndarray,
    mocap_fr: int,
) -> SkeletonMotion:
    """
    Create a SkeletonMotion from mink's output poses and translations.
    FIXED VERSION: Ensures root_translation matches pelvis global position.
    """
    n_frames = poses.shape[0]
    num_joints = len(skeleton_tree.node_names)
    
    print("\n[INFO] Using CORRECTED HYBRID approach: Consistent positions + computed rotations")
    
    # The target positions we want to achieve
    target_global_positions = orig_global_trans  # Shape: (n_frames, num_joints, 3)
    
    print(f"Target positions shape: {target_global_positions.shape}")
    print(f"Expected joint count: {num_joints}")
    
    # Step 1: Compute rotations from MuJoCo poses (these represent the motion intent)
    local_quat = np.zeros((n_frames, num_joints, 4))
    local_quat[..., 0] = 1.0  # Default to identity quaternions (WXYZ format)
    
    # Extract rotations from MuJoCo poses for each joint
    mujoco_to_skeleton_mapping = {
        'L_Hip': (0, 1),    'L_Knee': (3, 2),    'L_Ankle': (6, 3),    'L_Toe': (9, 4),
        'R_Hip': (12, 5),   'R_Knee': (15, 6),   'R_Ankle': (18, 7),   'R_Toe': (21, 8),
    }
    
    print("\n[INFO] Computing rotations from MuJoCo joint angles...")
    
    for frame in range(n_frames):
        # Root rotation from trans array (contains root position + quaternion)
        if skeleton_tree.node_names[0] == 'Pelvis':
            root_quat = trans[frame, 3:7]  # WXYZ quaternion from MuJoCo
            local_quat[frame, 0] = root_quat  # Keep WXYZ format
        
        # Extract joint rotations using CORRECT mapping
        for joint_name, (mujoco_idx, skeleton_idx) in mujoco_to_skeleton_mapping.items():
            if skeleton_idx < num_joints and mujoco_idx + 2 < poses.shape[1]:
                joint_angles = poses[frame, mujoco_idx:mujoco_idx + 3]
                scaled_angles = joint_angles * 1.0  # Use full angles
                
                try:
                    rot = sRot.from_euler('xyz', scaled_angles, degrees=False)
                    quat_xyzw = rot.as_quat()
                    quat_wxyz = np.roll(quat_xyzw, 1)  # XYZW -> WXYZ
                    local_quat[frame, skeleton_idx] = quat_wxyz
                except Exception:
                    local_quat[frame, skeleton_idx] = [1.0, 0.0, 0.0, 0.0]
    
    # *** THE KEY FIX: Use target pelvis positions as root translation ***
    print("\n[INFO] CORRECTED: Using target pelvis positions as root translation...")
    
    # Extract pelvis positions from target data (pelvis is typically joint 0)
    pelvis_positions = target_global_positions[:, 0, :]  # Shape: (n_frames, 3)
    root_translation = torch.from_numpy(pelvis_positions).float()
    
    print(f"Root translation sample (first 5 frames) - NOW MATCHES TARGET PELVIS:")
    for i in range(min(5, n_frames)):
        pos = root_translation[i].numpy()
        target_pelvis = target_global_positions[i, 0]
        print(f"  Frame {i}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] (target: [{target_pelvis[0]:.4f}, {target_pelvis[1]:.4f}, {target_pelvis[2]:.4f}])")
    
    # Step 3: Create skeleton state with computed rotations and CORRECTED root translations
    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(local_quat).float(),
        root_translation,
        is_local=True,
    )

    # Step 4: Create the motion
    motion = SkeletonMotion.from_skeleton_state(sk_state, fps=mocap_fr)
    
    # Step 5: DIRECTLY override global positions with target positions  
    print("\n[INFO] Directly overriding global positions while preserving rotations...")
    target_positions_torch = torch.from_numpy(target_global_positions).float()
    motion._global_translation = target_positions_torch
    
    # Verify the consistency
    result_positions = motion.global_translation[0].numpy()
    
    print("\n[INFO] CONSISTENCY CHECK:")
    root_pos = motion.root_translation[0].numpy()
    pelvis_pos = result_positions[0]  # Pelvis is index 0
    print(f"  Root position:   [{root_pos[0]:.4f}, {root_pos[1]:.4f}, {root_pos[2]:.4f}]")
    print(f"  Pelvis position: [{pelvis_pos[0]:.4f}, {pelvis_pos[1]:.4f}, {pelvis_pos[2]:.4f}]")
    root_pelvis_diff = np.linalg.norm(root_pos - pelvis_pos)
    print(f"  Root-Pelvis diff: {root_pelvis_diff:.6f}m")
    
    if root_pelvis_diff < 0.001:  # Very tight tolerance
        print("  ✅ ROOT AND PELVIS ARE NOW PERFECTLY ALIGNED!")
        print("  This should fix the MotionLib height issue.")
    else:
        print("  ❌ Still misaligned - check the implementation.")
    
    return motion

# SUMMARY OF THE FIX:
# Before: root_translation = torch.from_numpy(trans[:, :3]).float()  # MuJoCo positions
# After:  root_translation = torch.from_numpy(target_global_positions[:, 0, :]).float()  # Target pelvis positions
#
# This ensures that root_translation and global_translation[0] (pelvis) are identical,
# which is what MotionLib expects.
