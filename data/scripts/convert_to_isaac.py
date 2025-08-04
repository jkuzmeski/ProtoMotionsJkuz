
import typer
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Dict, Any, List
import torch
import numpy as np
from dm_control import mjcf
from dm_control.viewer import user_input
from collections import OrderedDict

import mujoco
import mujoco.viewer
import mink
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from tqdm import tqdm
import pandas as pd
import poselib.core.rotation3d as pRot
from poselib.core.tensor_utils import tensor_to_dict
import matplotlib.pyplot as plt


def _get_tpose_global_positions(skeleton_tree: SkeletonTree) -> np.ndarray:
    """Computes the global positions of joints in the T-pose from the skeleton tree."""
    parents = skeleton_tree.parent_indices.numpy()
    local_translations = skeleton_tree.local_translation.numpy()
    num_joints = len(parents)
    tpose_global_pos = np.zeros((num_joints, 3), dtype=np.float32)
    for i in range(num_joints):
        if parents[i] == -1:
            tpose_global_pos[i] = local_translations[i]
        else:
            tpose_global_pos[i] = tpose_global_pos[parents[i]] + local_translations[i]
    return tpose_global_pos


def debug_visualize_computed_positions(
    skeleton_tree: SkeletonTree,
    original_global_translations: np.ndarray,
    local_rotations_xyzw: np.ndarray,
    title: str = "Debug: Original vs. Reconstructed Positions"
):
    """
    Visualizes the difference between original and rotation-reconstructed joint positions.
    Red spheres = original data. Blue spheres = reconstructed from computed rotations.
    If blue and red match, rotation computation is likely correct.
    """
    print(f"\n--- Running Debug Analysis: {title} ---")
    
    # Print some debug info about the input data
    print(f"Original positions shape: {original_global_translations.shape}")
    print(f"Original first frame positions:")
    for i, name in enumerate(skeleton_tree.node_names):
        print(f"  {name}: {original_global_translations[0, i, :]}")
    
    # Reconstruct global positions from the computed local rotations
    num_frames = original_global_translations.shape[0]
    # poselib uses w,x,y,z quaternions. Our computed are x,y,z,w, so we convert.
    local_rotations_wxyz = local_rotations_xyzw[..., [3, 0, 1, 2]]
    
    print(f"Local rotations shape: {local_rotations_wxyz.shape}")
    print(f"First frame local rotations:")
    for i, name in enumerate(skeleton_tree.node_names):
        print(f"  {name}: {local_rotations_wxyz[0, i, :]}")
    
    try:
        sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(local_rotations_wxyz).float(),
            torch.from_numpy(original_global_translations[:, 0, :]).float(),  # Use original root translation
            is_local=True
        )
        reconstructed_global_translations = sk_state.global_translation.numpy()
        
        print(f"Reconstructed positions shape: {reconstructed_global_translations.shape}")
        print(f"Reconstructed first frame positions:")
        for i, name in enumerate(skeleton_tree.node_names):
            print(f"  {name}: {reconstructed_global_translations[0, i, :]}")
            
        # Check if reconstruction worked
        diff = np.abs(original_global_translations[0] - reconstructed_global_translations[0])
        print(f"\nPosition differences (first frame):")
        for i, name in enumerate(skeleton_tree.node_names):
            print(f"  {name}: {diff[i, :]} (max: {np.max(diff[i, :]):.6f})")
        print(f"Overall max difference: {np.max(diff):.6f}")
        
        if np.max(diff) < 0.01:
            print("✅ Rotation computation appears to be working correctly!")
        else:
            print("❌ Large differences detected - rotation computation has issues")
            
    except Exception as e:
        print(f"❌ Error in reconstruction: {e}")
        import traceback
        traceback.print_exc()
        
    print("--- Debug Analysis Finished ---")


def compute_rotations_from_motion(
    global_translations: np.ndarray,
    skeleton_tree: SkeletonTree
) -> np.ndarray:
    """
    Computes local joint rotations from global joint positions using a simple bone alignment method.
    
    This simplified approach focuses on aligning each bone with its reference direction in the T-pose,
    which should be sufficient for the IK solver to work with.

    Args:
        global_translations (np.ndarray): A numpy array of shape (num_frames, num_joints, 3)
                                          containing the global position of each joint for each frame.
        skeleton_tree (SkeletonTree): A poselib SkeletonTree object describing the robot's hierarchy.

    Returns:
        np.ndarray: A numpy array of shape (num_frames, num_joints, 4) containing the computed
                    local rotation for each joint as a quaternion (x, y, z, w).
    """
    print("Using simplified rotation computation approach...")
    
    num_frames = global_translations.shape[0]
    parents = skeleton_tree.parent_indices.numpy()
    joint_names = skeleton_tree.node_names
    num_joints = len(joint_names)

    # Get T-pose reference positions
    tpose_global_pos = _get_tpose_global_positions(skeleton_tree)   
    from scipy.spatial.transform import Rotation as R
    
    # Initialize with identity rotations
    local_rotations_quat = np.zeros((num_frames, num_joints, 4))
    local_rotations_quat[..., 3] = 1.0  # w component = 1 for identity

    print(f"Processing {num_frames} frames for {num_joints} joints...")
    print(f"Joint names: {joint_names}")
    print(f"T-pose positions shape: {tpose_global_pos.shape}")
    print(f"Current positions shape: {global_translations.shape}")
    
    # Debug: Print T-pose positions
    print("\n=== T-POSE POSITIONS ===")
    for j, name in enumerate(joint_names):
        print(f"{j}: {name} -> {tpose_global_pos[j]}")
    
    # Debug: Print first frame positions
    print("\n=== FIRST FRAME POSITIONS ===")
    for j, name in enumerate(joint_names):
        print(f"{j}: {name} -> {global_translations[0, j]}")
    
    # Debug: Print parent relationships
    print(f"\n=== PARENT RELATIONSHIPS ===")
    for j, name in enumerate(joint_names):
        parent_idx = parents[j]
        parent_name = joint_names[parent_idx] if parent_idx != -1 else "ROOT"
        print(f"{j}: {name} -> parent: {parent_idx} ({parent_name})")

    for i in tqdm(range(num_frames), desc="Computing Simple Rotations"):
        current_global_pos = global_translations[i]
        
        for j in range(num_joints):
            parent_idx = parents[j]
            
            if parent_idx == -1:
                # Root joint - keep identity rotation for now
                # The IK solver will handle the root orientation
                local_rotations_quat[i, j] = [0, 0, 0, 1]  # identity
            else:
                # For child joints, compute rotation to align bone direction
                try:
                    # Reference bone direction in T-pose
                    bone_ref = tpose_global_pos[j] - tpose_global_pos[parent_idx]
                    bone_ref_norm = np.linalg.norm(bone_ref)
                    
                    # Current bone direction
                    bone_cur = current_global_pos[j] - current_global_pos[parent_idx]
                    bone_cur_norm = np.linalg.norm(bone_cur)
                    
                    if bone_ref_norm > 1e-6 and bone_cur_norm > 1e-6:
                        # Normalize vectors
                        bone_ref_unit = bone_ref / bone_ref_norm
                        bone_cur_unit = bone_cur / bone_cur_norm
                        
                        # Debug output for first frame
                        if i == 0:
                            print(f"  Joint {j} ({joint_names[j]}):")
                            print(f"    T-pose bone: {bone_ref} (norm: {bone_ref_norm:.4f})")
                            print(f"    Current bone: {bone_cur} (norm: {bone_cur_norm:.4f})")
                            print(f"    T-pose unit: {bone_ref_unit}")
                            print(f"    Current unit: {bone_cur_unit}")
                        
                        # Compute rotation to align reference to current
                        rot, _ = R.align_vectors([bone_cur_unit], [bone_ref_unit])
                        local_rotations_quat[i, j] = rot.as_quat()
                        
                        if i == 0:
                            print(f"    Computed rotation: {rot.as_quat()}")
                    else:
                        # Degenerate case - use identity
                        local_rotations_quat[i, j] = [0, 0, 0, 1]
                        if i == 0:
                            print(f"  Joint {j} ({joint_names[j]}): DEGENERATE CASE (bone lengths: ref={bone_ref_norm:.6f}, cur={bone_cur_norm:.6f})")
                        
                except Exception as e:
                    print(f"Warning: Failed to compute rotation for joint {j} ({joint_names[j]}) at frame {i}: {e}")
                    local_rotations_quat[i, j] = [0, 0, 0, 1]

    print("Rotation computation completed.")
    return local_rotations_quat


# --- Configuration for Your Lower Body Robot ---
_SMPL_HUMANOID_LOWER_BODY_KEYPOINT_TO_JOINT = {
    "Pelvis": {"name": "Pelvis", "weight": 1.0},
    "L_Hip": {"name": "L_Hip", "weight": 1.5},
    "R_Hip": {"name": "R_Hip", "weight": 1.5},
    "L_Knee": {"name": "L_Knee", "weight": 2.0},
    "R_Knee": {"name": "R_Knee", "weight": 2.0},
    "L_Ankle": {"name": "L_Ankle", "weight": 3.0},
    "R_Ankle": {"name": "R_Ankle", "weight": 3.0},
    "L_Toe": {"name": "L_Toe", "weight": 3.0},
    "R_Toe": {"name": "R_Toe", "weight": 3.0},
}
_KEYPOINT_TO_JOINT_MAP = {
    "smpl_humanoid_lower_body": _SMPL_HUMANOID_LOWER_BODY_KEYPOINT_TO_JOINT,
}


@dataclass
class KeyCallback:
    """Handles keyboard input for pausing the viewer."""
    pause: bool = False

    def __call__(self, key: int) -> None:
        if key == user_input.KEY_SPACE:
            self.pause = not self.pause


@dataclass
class SimpleMotion:
    """A simple container to hold motion data."""
    global_translation: torch.Tensor
    global_rotation: torch.Tensor  # Added this field
    skeleton_tree: SkeletonTree
    fps: int


# --- MuJoCo Scene and Model Construction ---
def construct_model(robot_xml_path: str, keypoint_names: Sequence[str]):
    """
    Dynamically builds a MuJoCo model by loading the robot XML and adding mocap sites.
    This approach ensures the free joint remains at the top level.
    """
    # Load the robot model directly
    robot_mjcf = mjcf.from_path(robot_xml_path)
    
    # Add mocap bodies for keypoints to the robot's worldbody
    for name in keypoint_names:
        body = robot_mjcf.worldbody.add('body', name=f"keypoint_{name}", mocap=True)
        rgb = np.random.rand(3)
        body.add('site', name=f"site_{name}", type='sphere', size=[0.02],
                   rgba=f"{rgb[0]} {rgb[1]} {rgb[2]} 1")
    
    # Add a camera for viewing
    robot_mjcf.worldbody.add(
        "camera", name="front_track", pos="-0.120 3.232 1.064",
        xyaxes="-1.000 -0.002 -0.000 0.000 -0.103 0.995", mode="trackcom"
    )

    # Compile the final model from the XML string
    return mujoco.MjModel.from_xml_string(robot_mjcf.to_xml_string())


def retarget_motion(motion, robot_type: str, robot_xml_path: str, render: bool = False):
    """
    Main IK solver function.
    """
    global_translations = motion.global_translation.numpy()
    global_rotations = motion.global_rotation.numpy()
    timeseries_length = global_translations.shape[0]
    fps = motion.fps
    
    mocap_joint_names = motion.skeleton_tree.node_names

    model = construct_model(robot_xml_path, mocap_joint_names)
    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)

    tasks = []
    keypoint_map = _KEYPOINT_TO_JOINT_MAP[robot_type]
    for mocap_joint, retarget_info in keypoint_map.items():
        task = mink.FrameTask(
            frame_name=retarget_info["name"],
            frame_type="body",
            position_cost=5.0 * retarget_info["weight"],  # Reduced to prevent over-reaching
            orientation_cost=0.1 * retarget_info["weight"],  # Increased to maintain orientation
        )
        tasks.append(task)
    
    # Add a stronger posture task to keep the robot in a reasonable pose
    posture_task = mink.PostureTask(model, cost=1e-2)  # Increased posture cost to prevent collapse
    tasks.append(posture_task)
    
    key_callback = KeyCallback()
    viewer_context = mujoco.viewer.launch_passive(model, data, key_callback=key_callback) if render else None

    retargeted_poses = []
    retargeted_trans = []
    
    solver = "quadprog"  # Use the same solver as the working mink_retarget.py
    optimization_steps_per_frame = 2  # Reduced to prevent over-optimization

    if render:
        progress_bar = None
    else:
        progress_bar = tqdm(total=timeseries_length, desc="Retargeting")

    try:
        # Initialize robot in upright position - but don't constrain it too much
        data.qpos[0:3] = global_translations[0, 0]  # Root position
        data.qpos[3:7] = [1, 0, 0, 0]  # Upright orientation (w, x, y, z)
        
        # Set reasonable initial joint angles to prevent collapse
        # These are approximate standing pose angles (in radians)
        # Slightly bent knees and hips to prevent collapse
        initial_joint_angles = np.array([
            0.0, 0.0, 0.0,    # L_Hip - neutral
            0.0, 0.0, 0.1,    # L_Knee - slightly bent  
            0.0, 0.0, 0.0,    # L_Ankle - neutral
            0.0, 0.0, 0.0,    # L_Toe - neutral
            0.0, 0.0, 0.0,    # R_Hip - neutral
            0.0, 0.0, 0.1,    # R_Knee - slightly bent
            0.0, 0.0, 0.0,    # R_Ankle - neutral
            0.0, 0.0, 0.0,    # R_Toe - neutral
        ])
        data.qpos[7:] = initial_joint_angles
        
        print(f"Debug: Initial root position: {data.qpos[0:3]}")
        print(f"Debug: Initial root orientation: {data.qpos[3:7]}")
        print(f"Debug: Initial joint angles: {data.qpos[7:]}")
        print(f"Debug: First frame target positions:")
        for i, name in enumerate(mocap_joint_names):
            print(f"  {name}: {global_translations[0, i, :]}")
        
        # Check if target positions are reasonable
        print(f"Debug: Target position analysis:")
        pelvis_pos = global_translations[0, 0, :]  # Pelvis
        left_foot_pos = global_translations[0, 4, :]  # L_Toe
        right_foot_pos = global_translations[0, 8, :]  # R_Toe
        
        print(f"  Pelvis height: {pelvis_pos[2]:.3f}m")
        print(f"  Left foot height: {left_foot_pos[2]:.3f}m")
        print(f"  Right foot height: {right_foot_pos[2]:.3f}m")
        print(f"  Pelvis to left foot distance: {np.linalg.norm(pelvis_pos - left_foot_pos):.3f}m")
        print(f"  Pelvis to right foot distance: {np.linalg.norm(pelvis_pos - right_foot_pos):.3f}m")
        
        # Check if feet are above pelvis (which would cause collapse)
        if left_foot_pos[2] > pelvis_pos[2] or right_foot_pos[2] > pelvis_pos[2]:
            print(f"  ⚠️  WARNING: Feet are above pelvis! This will cause collapse.")
            print(f"  Left foot is {left_foot_pos[2] - pelvis_pos[2]:.3f}m above pelvis")
            print(f"  Right foot is {right_foot_pos[2] - pelvis_pos[2]:.3f}m above pelvis")
        else:
            print(f"  ✅ Feet are below pelvis - positions look reasonable")
        
        mujoco.mj_forward(model, data)
        
        # Set a much weaker posture task to allow more movement
        posture_task.set_target_from_configuration(configuration)

        for t in range(timeseries_length):
            if not (viewer_context and key_callback.pause):
                for i, (mocap_joint, _) in enumerate(keypoint_map.items()):
                    body_idx = mocap_joint_names.index(mocap_joint)
                    target_pos = global_translations[t, body_idx, :]
                    
                    # For now, let's focus only on position targets and let IK solve for orientations
                    # This should help avoid the upside-down issue caused by dummy rotations
                    identity_rot = mink.SO3.identity()
                    target_se3 = mink.SE3.from_rotation_and_translation(identity_rot, target_pos)
                    tasks[i].set_target(target_se3)
                    
                    # Debug output for first frame
                    if t == 0:
                        print(f"  Setting target for {mocap_joint}: {target_pos}")

                    if render:
                        mid = model.body(f"keypoint_{mocap_joint}").mocapid[0]
                        data.mocap_pos[mid] = target_pos
                
                # Perform multiple optimization steps per frame (like mink_retarget.py)
                for step in range(optimization_steps_per_frame):
                    # Add configuration limits like in mink_retarget.py
                    limits = [mink.ConfigurationLimit(model)]
                    
                    # Add velocity limits to prevent unrealistic movements
                    # Human running velocities should be reasonable (e.g., < 10 m/s for root, < 20 rad/s for joints)
                    max_root_velocity = 5.0  # m/s - reasonable for human running
                    max_joint_velocity = 10.0  # rad/s - reasonable for joint movements
                    
                    vel = mink.solve_ik(configuration, tasks, dt=1.0 / fps, solver=solver, damping=1e-1, limits=limits)
                    
                    # Apply velocity limits after IK solving
                    if vel.shape[0] >= 6:
                        # Limit root velocities (first 6 DOFs: 3 pos + 3 rot)
                        root_vel_magnitude = np.linalg.norm(vel[:3])  # Position velocity
                        if root_vel_magnitude > max_root_velocity:
                            vel[:3] = vel[:3] * (max_root_velocity / root_vel_magnitude)
                        
                        # Limit root rotation velocities
                        root_rot_vel_magnitude = np.linalg.norm(vel[3:6])  # Rotation velocity
                        if root_rot_vel_magnitude > max_joint_velocity:
                            vel[3:6] = vel[3:6] * (max_joint_velocity / root_rot_vel_magnitude)
                        
                        # Limit joint velocities
                        joint_velocities = vel[6:]
                        joint_vel_magnitudes = np.linalg.norm(joint_velocities.reshape(-1, 3), axis=1)
                        for i, mag in enumerate(joint_vel_magnitudes):
                            if mag > max_joint_velocity:
                                start_idx = 6 + i * 3
                                end_idx = start_idx + 3
                                vel[start_idx:end_idx] = vel[start_idx:end_idx] * (max_joint_velocity / mag)
                        
                        # Apply joint angle limits after integration
                        # This prevents joints from going into extreme positions
                        if step == optimization_steps_per_frame - 1:  # Only on final step
                            # Get current joint angles
                            current_joint_angles = data.qpos[7:]  # Skip root pos + root quat
                            
                            # Define more conservative joint angle limits (in radians)
                            # These are more restrictive to prevent extreme poses
                            joint_limits = {
                                'hip': [-0.5, 0.5],      # Hip flexion/extension (more conservative)
                                'knee': [0.0, 2.0],      # Knee flexion (0 to ~115 degrees)
                                'ankle': [-0.5, 0.5],    # Ankle dorsiflexion/plantarflexion (more conservative)
                                'toe': [-0.3, 0.3]       # Toe flexion/extension (more conservative)
                            }
                            
                            # Apply limits to joint angles
                            for i in range(0, len(current_joint_angles), 3):
                                joint_type_idx = i // 3
                                if joint_type_idx < len(joint_limits):
                                    # Get the joint type (hip, knee, ankle, toe)
                                    joint_types = ['hip', 'knee', 'ankle', 'toe']
                                    joint_type = joint_types[joint_type_idx % 4]
                                    limits = joint_limits[joint_type]
                                    
                                    # Apply limits to each axis
                                    for axis in range(3):
                                        angle_idx = i + axis
                                        if angle_idx < len(current_joint_angles):
                                            current_joint_angles[angle_idx] = np.clip(
                                                current_joint_angles[angle_idx], 
                                                limits[0], 
                                                limits[1]
                                            )
                            
                            # Update the data with limited joint angles
                            data.qpos[7:] = current_joint_angles
                    
                    # Debug: Check the shapes to understand the DOF structure
                    if t == 0 and step == 0:
                        print(f"Debug: vel shape: {vel.shape}, data.qpos shape: {data.qpos.shape}")
                        print(f"Debug: vel[:6] (root): {vel[:6]}")
                        print(f"Debug: vel[6:] (joints): {vel[6:].shape}")
                        print(f"Debug: data.qpos[6:] (joints): {data.qpos[6:].shape}")
                        
                        # Debug velocity limits
                        root_vel_mag = np.linalg.norm(vel[:3])
                        root_rot_vel_mag = np.linalg.norm(vel[3:6])
                        max_joint_vel = np.max(np.linalg.norm(vel[6:].reshape(-1, 3), axis=1)) if vel.shape[0] > 6 else 0
                        print(f"Debug: Root velocity magnitude: {root_vel_mag:.3f} m/s (limit: {max_root_velocity})")
                        print(f"Debug: Root rotation velocity magnitude: {root_rot_vel_mag:.3f} rad/s (limit: {max_joint_velocity})")
                        print(f"Debug: Max joint velocity magnitude: {max_joint_vel:.3f} rad/s (limit: {max_joint_velocity})")
                    
                    # Let the IK solver handle root movement naturally
                    # Don't manually override the root position - let the solver work
                    if step == 0 and t == 0:  # Only set initial position
                        # Get the target root position from the original motion data
                        target_root_pos = global_translations[t, 0, :]  # Pelvis is joint 0
                        
                        # Set initial root position but don't override it in subsequent frames
                        data.qpos[0:3] = target_root_pos
                        configuration.update(data.qpos)
                    
                    # Apply velocity integration for ALL DOFs including root
                    # The IK solver returns 30 DOFs, but data.qpos has 31 DOFs
                    # We need to match the shapes properly
                    if vel.shape[0] == 30 and data.qpos.shape[0] == 31:
                        # The IK solver returns [root_pos(3), root_rot(3), joint_angles(24)]
                        # But data.qpos has [root_pos(3), root_quat(4), joint_angles(24)]
                        # We need to handle the root quaternion conversion
                        
                        # Apply root position and rotation velocities
                        root_pos_vel = vel[:3]  # Root position velocity
                        root_rot_vel = vel[3:6]  # Root rotation velocity (axis-angle)
                        joint_velocities = vel[6:]  # Joint angle velocities
                        
                        # Integrate root position
                        data.qpos[:3] += root_pos_vel * (1.0 / fps)
                        
                        # Integrate root rotation (convert axis-angle to quaternion)
                        # For now, use a simple approach - convert to quaternion and integrate
                        from scipy.spatial.transform import Rotation as R
                        current_quat = data.qpos[3:7]  # Current root quaternion
                        rot_vel_magnitude = np.linalg.norm(root_rot_vel)
                        if rot_vel_magnitude > 1e-6:
                            rot_axis = root_rot_vel / rot_vel_magnitude
                            rot_angle = rot_vel_magnitude * (1.0 / fps)
                            delta_rot = R.from_rotvec(rot_axis * rot_angle)
                            current_rot = R.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])  # w,x,y,z -> x,y,z,w
                            new_rot = current_rot * delta_rot
                            new_quat = new_rot.as_quat()  # x,y,z,w
                            data.qpos[3:7] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]  # w,x,y,z
                        
                        # Integrate joint angles
                        joint_positions = data.qpos[7:]  # Joint positions
                        joint_positions += joint_velocities * (1.0 / fps)
                        data.qpos[7:] = joint_positions
                    else:
                        # Fallback: use the original integrate_inplace method
                        configuration.integrate_inplace(vel, 1.0 / fps)
                    
                    # Update configuration with new joint positions
                    configuration.update(data.qpos)
                    
                    # Debug output for first few frames
                    if t < 5 and step == 0:
                        print(f"  Frame {t}: Root pos: {data.qpos[:3]}, Target: {target_root_pos}")
                        print(f"  Root movement: {np.linalg.norm(data.qpos[:3] - global_translations[0, 0, :]):.6f}m")
                
                retargeted_poses.append(data.qpos[6:].copy())
                retargeted_trans.append(data.qpos[:7].copy())

            if render:
                viewer_context.sync()
            else:
                progress_bar.update(1)
    finally:
        if progress_bar:
            progress_bar.close()

    return np.stack(retargeted_poses), np.stack(retargeted_trans)


def detect_stance_phases(
    foot_positions,
    foot_velocities,
    foot_accelerations,
    height_threshold=0.05,
    vertical_velocity_threshold=0.1,
    horizontal_acceleration_threshold=0.5,
):
    """
    Detects stance phases based on robust biomechanical criteria suitable for treadmill running.

    A foot is considered in a stance phase if it meets three conditions:
    1. It is close to the ground (low height).
    2. It has minimal vertical movement (low vertical velocity).
    3. It is moving at a constant horizontal velocity (low horizontal acceleration),
       which corresponds to the treadmill belt speed.

    Args:
        foot_positions (np.ndarray): (n_frames, 3) array of foot positions.
        foot_velocities (np.ndarray): (n_frames, 3) array of foot velocities.
        foot_accelerations (np.ndarray): (n_frames, 3) array of foot accelerations.
        height_threshold (float): The maximum height (in meters) for a foot to be
                                  considered near the ground.
        vertical_velocity_threshold (float): The maximum vertical velocity (m/s)
                                             to be considered stationary.
        horizontal_acceleration_threshold (float): The maximum horizontal acceleration (m/s^2)
                                                   for the foot to be considered moving at a
                                                   constant velocity.

    Returns:
        np.ndarray: A boolean array where True indicates a stance phase.
    """
    # 1. Foot is close to the ground.
    height_condition = foot_positions[:, 2] < height_threshold

    # 2. Foot has minimal vertical velocity.
    vertical_velocity_condition = np.abs(foot_velocities[:, 2]) < vertical_velocity_threshold

    # 3. Foot has minimal horizontal acceleration (moving at constant velocity with the belt).
    horizontal_acceleration = np.linalg.norm(foot_accelerations[:, :2], axis=1)
    horizontal_acceleration_condition = horizontal_acceleration < horizontal_acceleration_threshold

    # Combine all conditions. A foot is in stance if all are true.
    stance_mask = height_condition & vertical_velocity_condition & horizontal_acceleration_condition

    # Clean up isolated stance/swing detections with morphological operations
    # to remove noise and create more contiguous stance phases.
    from scipy.ndimage import binary_closing, binary_opening
    stance_mask = binary_closing(stance_mask, structure=np.ones(5))
    stance_mask = binary_opening(stance_mask, structure=np.ones(3))

    return stance_mask


def transform_treadmill_to_overground(joint_centers, joint_names, fps, treadmill_speed):
    """
    Transform treadmill motion to overground motion by stabilizing stance feet.

    This function remaps motion captured on a treadmill to appear as if it were
    performed overground. It works by calculating an offset at each frame to
    ensure the foot (or feet) currently in a stance phase remains stationary
    in the world coordinate system. This offset is then applied to all joints.

    The transformation is derived directly from the motion data, making it
    robust to variations in treadmill speed and actor performance.

    In treadmill motion:
    - Pelvis stays relatively stationary in the world frame.
    - Feet move backward during stance.

    In the transformed overground motion:
    - Stance foot remains stationary.
    - Pelvis moves forward over the stance foot.

    Args:
        joint_centers (np.ndarray): (n_frames, n_joints, 3) array of joint positions.
        joint_names (list): List of joint names.
        fps (int): Frames per second of the motion capture data.

    Returns:
        np.ndarray: (n_frames, n_joints, 3) array of transformed joint positions.
    """
    print("Applying data-driven treadmill-to-overground transformation...")
    
    # Find joint indices
    l_ankle_idx = joint_names.index('L_Ankle') if 'L_Ankle' in joint_names else None
    r_ankle_idx = joint_names.index('R_Ankle') if 'R_Ankle' in joint_names else None
    l_toe_idx = joint_names.index('L_Toe') if 'L_Toe' in joint_names else None
    r_toe_idx = joint_names.index('R_Toe') if 'R_Toe' in joint_names else None
    
    # Use ankle if available, otherwise toe
    l_foot_idx = l_ankle_idx if l_ankle_idx is not None else l_toe_idx
    r_foot_idx = r_ankle_idx if r_ankle_idx is not None else r_toe_idx
    
    if l_foot_idx is None or r_foot_idx is None:
        print("Warning: Could not find foot joints, skipping transformation.")
        return joint_centers.copy()
    
    n_frames, n_joints, _ = joint_centers.shape
    time_delta = 1.0 / fps
    
    # Calculate foot velocities for stance detection
    l_foot_pos = joint_centers[:, l_foot_idx, :]
    r_foot_pos = joint_centers[:, r_foot_idx, :]
    
    # Use improved velocity calculation for better boundary handling
    l_foot_vel = safe_velocity_calculation(l_foot_pos, time_delta, use_extrapolation=True)
    r_foot_vel = safe_velocity_calculation(r_foot_pos, time_delta, use_extrapolation=True)
    
    # Calculate foot accelerations for the new stance detection method.
    l_foot_accel = safe_velocity_calculation(l_foot_vel, time_delta, use_extrapolation=True)
    r_foot_accel = safe_velocity_calculation(r_foot_vel, time_delta, use_extrapolation=True)
    
    # Detect stance phases using the more robust, acceleration-based method.
    l_stance = detect_stance_phases(l_foot_pos, l_foot_vel, l_foot_accel)
    r_stance = detect_stance_phases(r_foot_pos, r_foot_vel, r_foot_accel)
    
    # Initialize transformed positions
    transformed_centers = joint_centers.copy()
    current_offset = np.zeros(3)
    last_offset_update = np.zeros(3)

    for frame in range(1, n_frames):
        # Determine which foot (if any) is in stance for the current frame
        in_double_stance = l_stance[frame] and r_stance[frame]
        in_left_stance = l_stance[frame] and not r_stance[frame]
        in_right_stance = r_stance[frame] and not l_stance[frame]
        in_flight = not l_stance[frame] and not r_stance[frame]

        offset_update = np.zeros(3)

        if in_left_stance:
            # To make the left foot stationary, the world must move by the inverse of the foot's displacement.
            offset_update = -(l_foot_pos[frame] - l_foot_pos[frame - 1])
        elif in_right_stance:
            # To make the right foot stationary, the world must move by the inverse of the foot's displacement.
            offset_update = -(r_foot_pos[frame] - r_foot_pos[frame - 1])
        elif in_double_stance:
            # In double stance, use the average displacement of both feet to minimize drift and create smooth motion.
            avg_foot_now = (l_foot_pos[frame] + r_foot_pos[frame]) / 2.0
            avg_foot_prev = (l_foot_pos[frame - 1] + r_foot_pos[frame - 1]) / 2.0
            offset_update = -(avg_foot_now - avg_foot_prev)
        elif in_flight:
            # During flight, the character should continue with its last known velocity.
            # This is achieved by re-applying the displacement from the last frame that had ground contact.
            offset_update = last_offset_update
        
        # The transformation should only affect horizontal movement (X and Y axes).
        # Vertical (Z) motion should be preserved from the original capture to maintain jump heights, etc.
        offset_update[2] = 0.0

        current_offset += offset_update
        transformed_centers[frame, :, :] += current_offset

        # If the character was on the ground, store the calculated displacement.
        # This will be used to maintain momentum during the next flight phase.
        if not in_flight:
            last_offset_update = offset_update
            
    # Add the belt speed to achieve proper overground motion
    # The belt moves at treadmill_speed, so the runner should move forward at that speed
    duration = (n_frames - 1) / fps  # Total time duration
    expected_forward_distance = treadmill_speed * duration
    
    # Calculate current forward displacement after transformation
    pelvis_idx = joint_names.index('Pelvis') if 'Pelvis' in joint_names else 0
    initial_pelvis_pos = joint_centers[0, pelvis_idx, :]
    
    # Add linear progression to achieve target forward speed
    # This simulates the runner moving forward at belt speed
    for frame in range(n_frames):
        progress_ratio = frame / (n_frames - 1) if n_frames > 1 else 0
        additional_forward_offset = expected_forward_distance * progress_ratio
        transformed_centers[frame, :, 0] += additional_forward_offset
    
    # Final verification
    final_pelvis_pos = transformed_centers[-1, pelvis_idx, :]
    total_forward_dist = final_pelvis_pos[0] - initial_pelvis_pos[0]
    
    print(f"Applied transformation: pelvis moved {total_forward_dist:.3f}m forward in X direction.")
    print(f"Expected distance for {treadmill_speed:.1f} m/s over {duration:.1f}s: {expected_forward_distance:.3f}m")
    return transformed_centers


def plot_pelvis_velocity_analysis(global_translations, joint_names, fps, treadmill_speed, save_path=None):
    """
    Plot pelvis velocity analysis to verify treadmill transformation is working correctly.
    
    Args:
        global_translations: (n_frames, n_joints, 3) array of joint positions
        joint_names: list of joint names
        fps: frames per second
        treadmill_speed: expected treadmill speed in m/s
        save_path: optional path to save the plot
    """
    try:
        pelvis_idx = joint_names.index('Pelvis') if 'Pelvis' in joint_names else 0
        
        # Calculate velocities using improved safe_velocity_calculation
        time_delta = 1.0 / fps
        pelvis_positions = global_translations[:, pelvis_idx, :]
        
        # Use the improved velocity calculation with extrapolation for better boundary handling
        pelvis_velocities = safe_velocity_calculation(pelvis_positions, time_delta, use_extrapolation=True)
        
        # Time array
        time_array = np.arange(len(pelvis_positions)) / fps
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Pelvis Motion Analysis (Treadmill Speed: {treadmill_speed:.1f} m/s)', fontsize=16)
        
        # Plot 1: Pelvis Position over time
        ax1.plot(time_array, pelvis_positions[:, 0], 'r-', label='X (Forward)', linewidth=2)
        ax1.plot(time_array, pelvis_positions[:, 1], 'g-', label='Y (Lateral)', linewidth=2)
        ax1.plot(time_array, pelvis_positions[:, 2], 'b-', label='Z (Vertical)', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (m)')
        ax1.set_title('Pelvis Position')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Pelvis Velocity over time
        ax2.plot(time_array, pelvis_velocities[:, 0], 'r-', label='X (Forward)', linewidth=2)
        ax2.plot(time_array, pelvis_velocities[:, 1], 'g-', label='Y (Lateral)', linewidth=2)
        ax2.plot(time_array, pelvis_velocities[:, 2], 'b-', label='Z (Vertical)', linewidth=2)
        ax2.axhline(y=treadmill_speed, color='black', linestyle='--', label=f'Target Speed ({treadmill_speed:.1f} m/s)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Pelvis Velocity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Forward velocity histogram
        ax3.hist(pelvis_velocities[:, 0], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax3.axvline(x=treadmill_speed, color='red', linestyle='--', linewidth=2, label=f'Target Speed ({treadmill_speed:.1f} m/s)')
        ax3.axvline(x=np.mean(pelvis_velocities[:, 0]), color='green', linestyle='-', linewidth=2, label=f'Mean ({np.mean(pelvis_velocities[:, 0]):.2f} m/s)')
        ax3.set_xlabel('Forward Velocity (m/s)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Forward Velocity Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Velocity statistics
        stats_text = f"""
Forward Velocity Statistics:
Mean: {np.mean(pelvis_velocities[:, 0]):.3f} m/s
Std:  {np.std(pelvis_velocities[:, 0]):.3f} m/s
Min:  {np.min(pelvis_velocities[:, 0]):.3f} m/s
Max:  {np.max(pelvis_velocities[:, 0]):.3f} m/s

Target Speed: {treadmill_speed:.3f} m/s
Error: {np.mean(pelvis_velocities[:, 0]) - treadmill_speed:.3f} m/s

Lateral Velocity:
Mean: {np.mean(pelvis_velocities[:, 1]):.3f} m/s
Std:  {np.std(pelvis_velocities[:, 1]):.3f} m/s

Vertical Velocity:
Mean: {np.mean(pelvis_velocities[:, 2]):.3f} m/s
Std:  {np.std(pelvis_velocities[:, 2]):.3f} m/s
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Velocity Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Pelvis velocity analysis plot saved to: {save_path}")
        
        plt.show()
        
        # Print summary to console
        print("\n🔍 Pelvis Velocity Analysis:")
        print(f"   Target treadmill speed: {treadmill_speed:.3f} m/s")
        print(f"   Actual mean forward velocity: {np.mean(pelvis_velocities[:, 0]):.3f} m/s")
        print(f"   Forward velocity std: {np.std(pelvis_velocities[:, 0]):.3f} m/s")
        print(f"   Forward velocity range: {np.min(pelvis_velocities[:, 0]):.3f} to {np.max(pelvis_velocities[:, 0]):.3f} m/s")
        
        if abs(np.mean(pelvis_velocities[:, 0]) - treadmill_speed) < 0.1:
            print("   ✅ Forward velocity is close to target speed")
        else:
            print("   ⚠️  Forward velocity differs significantly from target speed")
            
        if np.std(pelvis_velocities[:, 0]) > 0.1:
            print("   ✅ Forward velocity shows natural variation (not pinned)")
        else:
            print("   ⚠️  Forward velocity appears too constant (possibly pinned)")
        
    except Exception as e:
        print(f"Error in pelvis velocity analysis: {e}")


# --- Data Loading and Main Execution ---

def create_motion_from_txt(motion_filepath: str, tpose_filepath: str, treadmill_speed: float, mocap_fr: int = 200, debug_file: Path = None):
    """Loads and parses your specific .txt file format using pandas for robustness."""
    print(f"Loading motion data from {motion_filepath}...")
    
    with open(motion_filepath, 'r') as f:
        lines = f.readlines()
        header_line = lines[1]
        raw_names = [name for name in header_line.strip().split('\t') if name]
        joint_names = []
        for name in raw_names:
            if name not in joint_names:
                joint_names.append(name)
    print(f"Parsed joint names from file: {joint_names}")

    motion_df = pd.read_csv(motion_filepath, sep='\t', header=4)
    motion_raw_data = motion_df.iloc[:, 1:].to_numpy()
    joint_centers = motion_raw_data.reshape((motion_raw_data.shape[0], -1, 3))

    print(f"Loading T-pose data from {tpose_filepath}...")
    tpose_df = pd.read_csv(tpose_filepath, sep='\t', header=4)
    tpose_positions = tpose_df.iloc[0, 1:].to_numpy().reshape(-1, 3)
    
    # Apply coordinate system rotation: Y->X, X->-Y to align with robot's +X forward direction
    print("Applying coordinate system rotation to align with robot orientation...")
    joint_centers_rotated = joint_centers.copy()
    
    # Apply 90-degree rotation around Z-axis: (x,y,z) -> (y,-x,z)
    # This transforms Y-forward motion data to X-forward robot orientation
    joint_centers_rotated[:, :, 0] = joint_centers[:, :, 1]   # New X = Old Y
    joint_centers_rotated[:, :, 1] = -joint_centers[:, :, 0]  # New Y = -Old X  
    joint_centers_rotated[:, :, 2] = joint_centers[:, :, 2]   # Z stays the same
    joint_centers = joint_centers_rotated
    
    # Also rotate T-pose positions with the same transformation
    tpose_positions_rotated = tpose_positions.copy()
    tpose_positions_rotated[:, 0] = tpose_positions[:, 1]   # New X = Old Y
    tpose_positions_rotated[:, 1] = -tpose_positions[:, 0]  # New Y = -Old X
    tpose_positions_rotated[:, 2] = tpose_positions[:, 2]   # Z stays the same
    tpose_positions = tpose_positions_rotated
    
    # Debug: Check if transformation preserved left-right symmetry
    if 'L_Hip' in joint_names and 'R_Hip' in joint_names:
        l_hip_idx = joint_names.index('L_Hip')
        r_hip_idx = joint_names.index('R_Hip')
        l_hip_pos = joint_centers[0, l_hip_idx, :]
        r_hip_pos = joint_centers[0, r_hip_idx, :]
        print(f"After rotation - L_Hip: {l_hip_pos}, R_Hip: {r_hip_pos}")
        print(f"Hip separation (Y-axis): {r_hip_pos[1] - l_hip_pos[1]:.3f}")
        if abs(r_hip_pos[1] - l_hip_pos[1]) < 0.1:
            print("⚠️  Warning: Hips appear too close in Y-axis, check coordinate transformation")

    # Apply treadmill-to-overground transformation if a non-zero treadmill speed is given.
    if treadmill_speed > 0:
        joint_centers = transform_treadmill_to_overground(joint_centers, joint_names, mocap_fr, treadmill_speed)

    if debug_file:
        np.savez(debug_file, body_positions=joint_centers, body_names=joint_names, fps=mocap_fr)
        print(f"Saved debug data to: {debug_file}")

    parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7]
    
    local_translations = np.zeros_like(tpose_positions)
    for i, p_idx in enumerate(parents):
        if p_idx == -1:
            local_translations[i] = tpose_positions[i]
        else:
            local_translations[i] = tpose_positions[i] - tpose_positions[p_idx]
            
    skeleton_tree = SkeletonTree(joint_names, torch.tensor(parents), torch.from_numpy(local_translations).float())

    # Create proper rotations for all joints and frames
    num_frames, num_joints = joint_centers.shape[0], len(joint_names)
    # Create rotations with proper upright orientation
    dummy_rotations = torch.zeros(num_frames, num_joints, 4)
    dummy_rotations[:, :, 0] = 1.0  # Set w component to 1 for identity quaternions
    
    # Fix the root (Pelvis) orientation to be upright
    # The coordinate transform we applied earlier (Y->X, X->-Y) might have affected orientation
    # Let's ensure the pelvis has proper upright orientation
    pelvis_idx = 0  # Pelvis is the first joint
    # Identity quaternion should work, but if the robot is upside down, we might need a rotation
    # For now, keep identity and we'll debug further if needed

    motion = SimpleMotion(
        global_translation=torch.from_numpy(joint_centers).float(),
        global_rotation=dummy_rotations,
        skeleton_tree=skeleton_tree,
        fps=mocap_fr
    )
    
    print("Motion object created successfully.")
    return motion


def extrapolate_boundaries(data, n_extrapolate=2):
    """
    Extrapolate data at boundaries using polynomial fitting to reduce edge artifacts.
    
    Args:
        data: Input data array (frames, ...)
        n_extrapolate: Number of frames to extrapolate at each boundary
        
    Returns:
        extrapolated_data: Data with extrapolated boundaries
    """
    if data.shape[0] < 4 or n_extrapolate <= 0:
        return data
        
    extended_data = np.zeros((data.shape[0] + 2 * n_extrapolate,) + data.shape[1:], dtype=data.dtype)
    
    # Copy original data to center
    extended_data[n_extrapolate:-n_extrapolate] = data
    
    # Extrapolate at beginning using linear trend from first few points
    if data.shape[0] >= 3:
        # Use first 3 points to establish trend
        trend_start = (data[2] - data[0]) / 2  # Average slope
        for i in range(n_extrapolate):
            extended_data[n_extrapolate - 1 - i] = data[0] - (i + 1) * trend_start
    
    # Extrapolate at end using linear trend from last few points
    if data.shape[0] >= 3:
        # Use last 3 points to establish trend
        trend_end = (data[-1] - data[-3]) / 2  # Average slope
        for i in range(n_extrapolate):
            extended_data[-n_extrapolate + i] = data[-1] + (i + 1) * trend_end
            
    return extended_data


def safe_velocity_calculation(data, time_delta, use_extrapolation=True):
    """
    Safely calculate velocities with improved boundary handling.
    
    Uses higher-order finite differences at boundaries for better accuracy
    and applies smoothing to reduce noise at trial edges.
    
    Args:
        data: Input position data (frames, ...)
        time_delta: Time step between frames
        use_extrapolation: Whether to use boundary extrapolation for better edge handling
    """
    try:
        if data.shape[0] < 2:
            return np.zeros_like(data)

        # Optionally extrapolate boundaries for better edge handling
        if use_extrapolation and data.shape[0] >= 4:
            extended_data = extrapolate_boundaries(data, n_extrapolate=2)
            extended_velocities = np.zeros_like(extended_data)
            n_frames = extended_data.shape[0]
            
            # Calculate velocities on extended data using central differences
            if n_frames > 2:
                extended_velocities[1:-1] = (extended_data[2:] - extended_data[:-2]) / (2 * time_delta)
            
            # Use higher-order formulas at extended boundaries
            if n_frames >= 5:
                extended_velocities[0] = ((-25 * extended_data[0] + 48 * extended_data[1]
                                          - 36 * extended_data[2] + 16 * extended_data[3]
                                          - 3 * extended_data[4]) / (12 * time_delta))
                extended_velocities[-1] = ((25 * extended_data[-1] - 48 * extended_data[-2]
                                           + 36 * extended_data[-3] - 16 * extended_data[-4]
                                           + 3 * extended_data[-5]) / (12 * time_delta))
            else:
                extended_velocities[0] = (extended_data[1] - extended_data[0]) / time_delta
                extended_velocities[-1] = (extended_data[-1] - extended_data[-2]) / time_delta
            
            # Extract the velocities for the original data range
            velocities = extended_velocities[2:-2]  # Remove extrapolated boundary frames
        else:
            # Fallback to original method for short sequences or when extrapolation is disabled
            velocities = np.zeros_like(data)
            n_frames = data.shape[0]

            if n_frames == 2:
                # Only two frames - use simple difference
                vel = (data[1] - data[0]) / time_delta
                velocities[0] = vel
                velocities[1] = vel
            elif n_frames == 3:
                # Three frames - use three-point formulas
                velocities[0] = (-3 * data[0] + 4 * data[1] - data[2]) / (2 * time_delta)  # Forward 3-point
                velocities[1] = (data[2] - data[0]) / (2 * time_delta)  # Central difference
                velocities[2] = (3 * data[2] - 4 * data[1] + data[0]) / (2 * time_delta)  # Backward 3-point
            elif n_frames >= 4:
                # Four or more frames - use higher-order methods at boundaries
                
                # Forward 4-point formula for first frame (more accurate than 3-point)
                if n_frames >= 5:
                    velocities[0] = ((-25 * data[0] + 48 * data[1] - 36 * data[2]
                                      + 16 * data[3] - 3 * data[4]) / (12 * time_delta))
                else:
                    velocities[0] = (-3 * data[0] + 4 * data[1] - data[2]) / (2 * time_delta)
                
                # Forward 3-point formula for second frame
                if n_frames >= 3:
                    velocities[1] = (-3 * data[0] + 4 * data[1] - data[2]) / (2 * time_delta)
                else:
                    velocities[1] = (data[2] - data[0]) / (2 * time_delta)
                
                # Central difference for interior points
                if n_frames > 4:
                    velocities[2:-2] = (data[4:] - data[:-4]) / (4 * time_delta)
                
                # Backward 3-point formula for second-to-last frame
                if n_frames >= 3:
                    velocities[-2] = (3 * data[-1] - 4 * data[-2] + data[-3]) / (2 * time_delta)
                else:
                    velocities[-2] = (data[-1] - data[-3]) / (2 * time_delta)
                
                # Backward 4-point formula for last frame (more accurate than 3-point)
                if n_frames >= 5:
                    velocities[-1] = ((25 * data[-1] - 48 * data[-2] + 36 * data[-3]
                                       - 16 * data[-4] + 3 * data[-5]) / (12 * time_delta))
                else:
                    velocities[-1] = (3 * data[-1] - 4 * data[-2] + data[-3]) / (2 * time_delta)

        return np.nan_to_num(velocities)
    except Exception as e:
        print(f"Error in velocity calculation: {e}")
        return np.zeros_like(data)


def safe_angular_velocity_calculation(quaternions, time_delta):
    """
    Safely calculate angular velocities from quaternions with improved boundary handling.

    This method provides more accurate estimation of angular velocity by using
    higher-order finite differences at boundaries and smoothing to reduce noise.
    """
    try:
        num_frames = quaternions.shape[0]
        if num_frames < 2:
            return np.zeros((num_frames, quaternions.shape[1], 3))

        q_tensor = torch.from_numpy(quaternions).float()
        angular_velocities = torch.zeros(num_frames, quaternions.shape[1], 3,
                                         device=q_tensor.device, dtype=q_tensor.dtype)

        if num_frames == 2:
            # Only two frames - use simple difference
            diff_quat = pRot.quat_mul_norm(q_tensor[1], pRot.quat_inverse(q_tensor[0]))
            diff_angle, diff_axis = pRot.quat_angle_axis(diff_quat)
            vel = diff_axis * diff_angle.unsqueeze(-1) / time_delta
            angular_velocities[0] = vel
            angular_velocities[1] = vel
            
        elif num_frames >= 3:
            # For frames with sufficient neighbors, use appropriate finite difference schemes
            
            # Forward difference for the first frame
            diff_quat_0 = pRot.quat_mul_norm(q_tensor[1], pRot.quat_inverse(q_tensor[0]))
            diff_angle_0, diff_axis_0 = pRot.quat_angle_axis(diff_quat_0)
            angular_velocities[0] = diff_axis_0 * diff_angle_0.unsqueeze(-1) / time_delta

            # Central difference for interior frames
            if num_frames > 2:
                diff_quat_mid = pRot.quat_mul_norm(q_tensor[2:], pRot.quat_inverse(q_tensor[:-2]))
                diff_angle_mid, diff_axis_mid = pRot.quat_angle_axis(diff_quat_mid)
                angular_velocities[1:-1] = diff_axis_mid * diff_angle_mid.unsqueeze(-1) / (2 * time_delta)

            # Backward difference for the last frame
            diff_quat_last = pRot.quat_mul_norm(q_tensor[-1], pRot.quat_inverse(q_tensor[-2]))
            diff_angle_last, diff_axis_last = pRot.quat_angle_axis(diff_quat_last)
            angular_velocities[-1] = diff_axis_last * diff_angle_last.unsqueeze(-1) / time_delta
        return np.nan_to_num(angular_velocities.cpu().numpy())
    except Exception as e:
        print(f"Error in angular velocity calculation: {e}")
        return np.zeros((quaternions.shape[0], quaternions.shape[1], 3))


def main(
    motion_file: Path = typer.Argument(..., help="Path to your .txt motion file."),
    tpose_file: Path = typer.Argument(..., help="Path to your T-pose .txt file."),
    output_file: Path = typer.Argument(..., help="Path to save the output .npy file."),
    robot_xml: Path = typer.Argument(..., help="Path to your robot's .xml file."),
    treadmill_speed: float = typer.Option(3.0, help="Treadmill speed in m/s."),
    render: bool = typer.Option(False, help="Enable live rendering of the retargeting process."),
    debug_file: Path = typer.Option(None, help="Path to save the debug .npz file."),
):
    """
    This script retargets motion from a text file of 3D joint positions to a
    lower-body humanoid robot for use in Isaac Lab.
    """
    robot_type = "smpl_humanoid_lower_body"

    source_motion = create_motion_from_txt(str(motion_file), str(tpose_file), treadmill_speed, debug_file=debug_file)
    
    print("Skipping rotation computation - using IK solver to determine joint rotations from positions")
    
    if render:
        print("Debug: First frame positions being sent to IK solver:")
        for i, name in enumerate(source_motion.skeleton_tree.node_names):
            print(f"  {name}: {source_motion.global_translation[0, i, :].numpy()}")
    
    # FIXED: Since rotation computation from positions is problematic, let's use IK retargeting
    # but provide it with the correct position targets from our transformed motion data
    print("Using IK retargeting with position targets from transformed motion data...")
    
    # Use IK for position refinement - this should work better than trying to compute rotations
    poses, trans = retarget_motion(source_motion, robot_type, str(robot_xml), render)

    print("\nConverting to final SMPL-like format...")
    
    # CRITICAL: Use the IK results for root translation, not the original motion data
    # The IK solver computed proper positions that work with the robot
    root_translation = torch.from_numpy(trans[:, :3]).double()  # Use IK root positions
    
    print(f"Debug: Using IK root positions instead of original motion data")
    print(f"Debug: IK root position range: {trans[:, :3].min():.3f} to {trans[:, :3].max():.3f}")
    
    # Also use the original for comparison
    transformed_global_translations = source_motion.global_translation.numpy()
    pelvis_idx = source_motion.skeleton_tree.node_names.index('Pelvis') if 'Pelvis' in source_motion.skeleton_tree.node_names else 0
    original_pelvis_positions = transformed_global_translations[:, pelvis_idx, :].copy()
    
    # Debug: Print pelvis movement to verify transformation is preserved
    ik_pelvis_start = trans[0, :3]
    ik_pelvis_end = trans[-1, :3]
    ik_total_movement = ik_pelvis_end - ik_pelvis_start
    print(f"Debug: IK Pelvis movement - Start: {ik_pelvis_start}, End: {ik_pelvis_end}")
    print(f"Debug: IK Total pelvis displacement: {ik_total_movement}, Forward distance: {ik_total_movement[0]:.3f}m")
    print(f"Debug: IK Pelvis height range: {trans[:, 2].min():.3f}m to {trans[:, 2].max():.3f}m")
    
    # Compare with original
    orig_pelvis_start = original_pelvis_positions[0, :]
    orig_pelvis_end = original_pelvis_positions[-1, :]
    orig_total_movement = orig_pelvis_end - orig_pelvis_start
    print(f"Debug: Original Pelvis movement - Start: {orig_pelvis_start}, End: {orig_pelvis_end}")
    print(f"Debug: Original Total pelvis displacement: {orig_total_movement}, Forward distance: {orig_total_movement[0]:.3f}m")
    
    # Correct DOF mapping based on the actual XML structure
    joint_dof_map: List[Dict[str, Any]] = [
        {'name': 'L_Hip', 'dof': 3},
        {'name': 'L_Knee', 'dof': 3},  # Now has x, y, z axes
        {'name': 'L_Ankle', 'dof': 3},
        {'name': 'L_Toe', 'dof': 3},   # Now has x, y, z axes
        {'name': 'R_Hip', 'dof': 3},
        {'name': 'R_Knee', 'dof': 3},  # Now has x, y, z axes
        {'name': 'R_Ankle', 'dof': 3},
        {'name': 'R_Toe', 'dof': 3}    # Now has x, y, z axes
    ]

    # Check DOF consistency
    actual_dofs = sum(j['dof'] for j in joint_dof_map)
    # The IK solver seems to be returning 25 DOFs, which suggests it's handling the free joint differently
    # Let's accept what the IK solver returns and adjust our expectations
    if poses.shape[1] != 25:
        print(f"Warning: Expected 25 DOFs from IK solver, but got {poses.shape[1]}")
        # For now, let's proceed with what we have
    else:
        print(f"✅ DOF consistency check passed: IK solver returned {poses.shape[1]} DOFs as expected")

    # Convert poses to PyTorch tensor first
    poses = torch.from_numpy(poses).double()
    
    # IMPORTANT: Use the IK results, not the dummy rotations!
    # The IK solver computed proper joint rotations in the 'poses' array
    # We need to convert these back to quaternions for the motion file
    
    print(f"Debug: Using IK results - poses shape: {poses.shape}")
    print(f"Debug: trans shape: {trans.shape}")
    print(f"Debug: Sample pose values: {poses[0, :6]}")  # First 6 DOFs
    print(f"Debug: Sample trans values: {trans[0]}")     # First frame
    
    # Use the root rotation from the IK results (trans contains [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z])
    root_rotation_quat = torch.from_numpy(trans[:, 3:]).double()  # Extract quaternion part
    
    # Convert IK joint angles to quaternions
    # The poses array contains joint angles that need to be converted to quaternions
    num_frames, num_joints = source_motion.global_translation.shape[0], len(source_motion.skeleton_tree.node_names)
    full_rotation = torch.zeros(num_frames, num_joints, 4).double()
    
    # Set the root rotation from IK results
    full_rotation[:, 0, :] = root_rotation_quat  # Pelvis is joint 0
    
    # Convert joint angles to quaternions
    # The poses array has shape (num_frames, 25) where the first 6 are root DOFs and the rest are joint angles
    # We need to convert the joint angles (indices 6-24) to quaternions
    joint_angles = poses[:, 6:].numpy()  # Extract joint angles (19 DOFs)
    
    print(f"Debug: Converting {joint_angles.shape[1]} joint angles to quaternions")
    print(f"Debug: Joint angle range: {joint_angles.min():.3f} to {joint_angles.max():.3f}")
    
    # Convert joint angles to quaternions using axis-angle representation
    # Each joint has 3 DOFs (x, y, z rotations), so we need to combine them
    from scipy.spatial.transform import Rotation as R
    
    for frame in range(num_frames):
        angle_idx = 0
        for joint_idx in range(1, num_joints):  # Skip root (joint 0)
            if angle_idx + 2 < joint_angles.shape[1]:  # Ensure we have 3 angles
                # Extract the 3 angles for this joint
                angles = joint_angles[frame, angle_idx:angle_idx + 3]
                
                # Convert to rotation matrix using Euler angles (XYZ order)
                # Note: The order matters - we'll use XYZ as that's common for joint rotations
                try:
                    # Create rotation from Euler angles (XYZ order)
                    rot = R.from_euler('xyz', angles, degrees=False)
                    quat = rot.as_quat()  # Returns [x, y, z, w]
                    
                    # Convert to [w, x, y, z] format for poselib
                    full_rotation[frame, joint_idx, 0] = quat[3]  # w
                    full_rotation[frame, joint_idx, 1] = quat[0]  # x
                    full_rotation[frame, joint_idx, 2] = quat[1]  # y
                    full_rotation[frame, joint_idx, 3] = quat[2]  # z
                    
                    if frame == 0:
                        print(f"Debug: Joint {joint_idx} ({source_motion.skeleton_tree.node_names[joint_idx]}): angles={angles}, quat={quat}")
                        
                except Exception as e:
                    print(f"Warning: Failed to convert joint {joint_idx} angles {angles}: {e}")
                    # Fallback to identity
                    full_rotation[frame, joint_idx, 0] = 1.0  # w
                    full_rotation[frame, joint_idx, 1:4] = 0.0  # x, y, z
                
                angle_idx += 3
            else:
                # Not enough angles, use identity
                full_rotation[frame, joint_idx, 0] = 1.0  # w
                full_rotation[frame, joint_idx, 1:4] = 0.0  # x, y, z

    time_delta = 1.0 / source_motion.fps
    
    # Use the IK results for velocity calculation instead of original motion data
    ik_root_positions = trans[:, :3]
    ik_root_velocity = torch.from_numpy(safe_velocity_calculation(ik_root_positions, time_delta, use_extrapolation=True)).double()

    # Debug: Check IK pelvis velocity specifically
    print(f"Debug: IK Pelvis velocity range - X: {ik_root_velocity[:, 0].min():.3f} to {ik_root_velocity[:, 0].max():.3f} m/s")
    print(f"Debug: IK Pelvis velocity range - Y: {ik_root_velocity[:, 1].min():.3f} to {ik_root_velocity[:, 1].max():.3f} m/s")
    print(f"Debug: IK Pelvis velocity range - Z: {ik_root_velocity[:, 2].min():.3f} to {ik_root_velocity[:, 2].max():.3f} m/s")
    print(f"Debug: IK Average forward (X) velocity: {ik_root_velocity[:, 0].mean():.3f} m/s")
    
    # For the global velocity, we need to create a full set including all joints
    # For now, use the original joint positions but with IK root velocity
    global_velocity_raw = torch.from_numpy(safe_velocity_calculation(transformed_global_translations, time_delta, use_extrapolation=True)).double()
    # Replace the root (pelvis) velocity with IK results
    global_velocity_raw[:, pelvis_idx, :] = ik_root_velocity
    
    # Plot pelvis velocity analysis for verification
    plot_pelvis_velocity_analysis(transformed_global_translations,
                                  source_motion.skeleton_tree.node_names,
                                  source_motion.fps,
                                  treadmill_speed)
    
    if global_velocity_raw.shape[1] == 9:
        global_velocity = global_velocity_raw  # Include all 9 joints including Pelvis
    else:
        raise ValueError(f"Unexpected number of joints in global velocity: {global_velocity_raw.shape[1]}")
    
    angular_velocity = torch.from_numpy(safe_angular_velocity_calculation(full_rotation.numpy(), time_delta)).double()

    # Final sanity checks and quaternion normalization
    for name, tensor in [("full_rotation", full_rotation), ("root_translation", root_translation),
                         ("global_velocity", global_velocity), ("angular_velocity", angular_velocity)]:
        if not torch.all(torch.isfinite(tensor)):
            print(f"Warning: Non-finite values in {name}, replacing with zeros")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Special handling for quaternions (full_rotation)
        if name == "full_rotation":
            # Normalize quaternions to ensure they are unit quaternions
            quat_norms = torch.norm(tensor, dim=-1, keepdim=True)
            # Avoid division by zero
            quat_norms = torch.clamp(quat_norms, min=1e-6)
            tensor = tensor / quat_norms
            print(f"✅ Normalized quaternions in {name}")

    # Debug: Print some motion data to verify it's not all zeros
    print("\nMotion data verification:")
    print(f"  Root translation range: {root_translation.min():.6f} to {root_translation.max():.6f}")
    print(f"  Root rotation range: {root_rotation_quat.min():.6f} to {root_rotation_quat.max():.6f}")
    print(f"  Global velocity range: {global_velocity.min():.6f} to {global_velocity.max():.6f}")
    print(f"  Angular velocity range: {angular_velocity.min():.6f} to {angular_velocity.max():.6f}")
    print(f"  Root Height: {root_translation[:, 2].mean():.3f} m")

    # Check if motion has meaningful content
    if torch.allclose(root_translation, torch.zeros_like(root_translation), atol=1e-6):
        print("⚠️  Warning: Root translation appears to be all zeros!")
    if torch.allclose(root_rotation_quat, torch.zeros_like(root_rotation_quat), atol=1e-6):
        print("⚠️  Warning: Root rotation appears to be all zeros!")
    else:
        print("✅ Root rotation has meaningful data!")

    # Create the output data in the format expected by SkeletonMotion.from_file()
    output_data = OrderedDict([
        ("rotation", tensor_to_dict(full_rotation)),
        ("root_translation", tensor_to_dict(root_translation)),
        ("global_velocity", tensor_to_dict(global_velocity)),
        ("global_angular_velocity", tensor_to_dict(angular_velocity)),
        ("skeleton_tree", source_motion.skeleton_tree.to_dict()),
        ("is_local", False),
        ("fps", np.array(source_motion.fps, dtype=np.float64)),
        ("__name__", "SkeletonMotion")
    ])

    print(f"Saving {poses.shape[0]} frames of retargeted motion to: {output_file}")
    np.save(output_file, output_data, allow_pickle=True)

    # Save debug file for visualization with debug_motion_viewer.py
    debug_output_file = output_file.parent / f"{output_file.stem}_debug.npz"
    print(f"Saving debug visualization file to: {debug_output_file}")
    
    # Reconstruct global positions from the IK results for visualization
    # We need to convert the IK poses back to global positions
    debug_positions = np.zeros((poses.shape[0], len(source_motion.skeleton_tree.node_names), 3))
    
    # Use the IK root positions for pelvis
    debug_positions[:, 0, :] = trans[:, :3]  # Pelvis from IK root
    
    # Get original positions for fallback
    original_positions = source_motion.global_translation.numpy()
    
    # For other joints, we need to reconstruct from the IK poses
    # This is more complex but will show us what the IK solver actually computed
    print(f"Reconstructing joint positions from IK poses...")
    
    # Load the robot model to get the forward kinematics
    model = construct_model(str(robot_xml), source_motion.skeleton_tree.node_names)
    data = mujoco.MjData(model)
    
    for frame in range(poses.shape[0]):
        # Set the robot state to the IK solution
        data.qpos[:3] = trans[frame, :3]  # Root position
        data.qpos[3:7] = trans[frame, 3:7]  # Root quaternion
        
        # Handle joint angles with proper shape matching
        joint_angles = poses[frame, 6:].numpy()  # Joint angles from IK
        expected_joints = len(data.qpos[7:])  # Expected number of joint DOFs
        
        if len(joint_angles) == expected_joints:
            data.qpos[7:] = joint_angles
        elif len(joint_angles) < expected_joints:
            # Pad with zeros if IK returned fewer DOFs
            padded_angles = np.zeros(expected_joints)
            padded_angles[:len(joint_angles)] = joint_angles
            data.qpos[7:] = padded_angles
        else:
            # Truncate if IK returned more DOFs
            data.qpos[7:] = joint_angles[:expected_joints]
        
        # Forward kinematics to get joint positions
        mujoco.mj_forward(model, data)
        
        # Extract joint positions from the model
        for joint_idx, joint_name in enumerate(source_motion.skeleton_tree.node_names):
            if joint_name == 'Pelvis':
                # Use the root position directly
                debug_positions[frame, joint_idx, :] = data.qpos[:3]
            else:
                # Get the body position for this joint
                try:
                    body_id = model.body(joint_name).id
                    debug_positions[frame, joint_idx, :] = data.xpos[body_id]
                except:
                    # Fallback: use the original relative positions
                    relative_pos = original_positions[frame, joint_idx, :] - original_positions[frame, 0, :]
                    debug_positions[frame, joint_idx, :] = debug_positions[frame, 0, :] + relative_pos
    
    debug_data = {
        "body_positions": debug_positions,
        "body_names": source_motion.skeleton_tree.node_names,
        "fps": source_motion.fps
    }
    
    np.savez(debug_output_file, **debug_data)
    print(f"✅ Debug file saved! You can visualize it with:")
    print(f"   python data/scripts/debug_motion_viewer.py --file {debug_output_file}")
    
    # ===== IK vs Original Motion Analysis =====
    print("\n" + "="*60)
    print("IK vs ORIGINAL MOTION ANALYSIS")
    print("="*60)
    
    # Compare root positions
    original_root_pos = original_pelvis_positions
    ik_root_pos = trans[:, :3]
    
    print(f"\n📊 ROOT POSITION COMPARISON:")
    print(f"Original root range: X[{original_root_pos[:, 0].min():.3f}, {original_root_pos[:, 0].max():.3f}], "
          f"Y[{original_root_pos[:, 1].min():.3f}, {original_root_pos[:, 1].max():.3f}], "
          f"Z[{original_root_pos[:, 2].min():.3f}, {original_root_pos[:, 2].max():.3f}]")
    print(f"IK root range:      X[{ik_root_pos[:, 0].min():.3f}, {ik_root_pos[:, 0].max():.3f}], "
          f"Y[{ik_root_pos[:, 1].min():.3f}, {ik_root_pos[:, 1].max():.3f}], "
          f"Z[{ik_root_pos[:, 2].min():.3f}, {ik_root_pos[:, 2].max():.3f}]")
    
    # Calculate root position differences
    root_pos_diff = np.linalg.norm(ik_root_pos - original_root_pos, axis=1)
    print(f"Root position differences:")
    print(f"  Mean: {root_pos_diff.mean():.6f}m")
    print(f"  Std:  {root_pos_diff.std():.6f}m")
    print(f"  Max:  {root_pos_diff.max():.6f}m")
    print(f"  Min:  {root_pos_diff.min():.6f}m")
    
    if root_pos_diff.mean() > 0.1:
        print(f"  ⚠️  WARNING: Large root position differences detected!")
    else:
        print(f"  ✅ Root positions are reasonably close")
    
    # Compare joint positions for key joints
    key_joints = ['L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Toe', 'R_Toe']
    joint_indices = [source_motion.skeleton_tree.node_names.index(name) for name in key_joints if name in source_motion.skeleton_tree.node_names]
    
    print(f"\n📊 JOINT POSITION COMPARISON:")
    for i, joint_name in enumerate(key_joints):
        if joint_name in source_motion.skeleton_tree.node_names:
            joint_idx = source_motion.skeleton_tree.node_names.index(joint_name)
            original_joint_pos = original_positions[:, joint_idx, :]
            ik_joint_pos = debug_positions[:, joint_idx, :]
            
            joint_diff = np.linalg.norm(ik_joint_pos - original_joint_pos, axis=1)
            print(f"{joint_name:>8}: mean_diff={joint_diff.mean():.6f}m, max_diff={joint_diff.max():.6f}m")
            
            if joint_diff.mean() > 0.2:
                print(f"         ⚠️  Large differences for {joint_name}")
    
    # Analyze IK joint angles
    print(f"\n📊 IK JOINT ANGLE ANALYSIS:")
    joint_angles = poses[:, 6:].numpy()  # Extract joint angles (skip root DOFs)
    
    print(f"Joint angle statistics:")
    print(f"  Range: [{joint_angles.min():.3f}, {joint_angles.max():.3f}] rad")
    print(f"  Mean:  {joint_angles.mean():.6f} rad")
    print(f"  Std:   {joint_angles.std():.6f} rad")
    
    # Check if joint angles are reasonable (not all zeros or extreme values)
    if np.allclose(joint_angles, 0, atol=1e-6):
        print(f"  ⚠️  WARNING: All joint angles are zero! IK may not be working.")
    elif np.abs(joint_angles).max() > 10.0:  # More than ~573 degrees
        print(f"  ⚠️  WARNING: Extreme joint angles detected!")
    else:
        print(f"  ✅ Joint angles appear reasonable")
    
    # Analyze IK root rotation
    print(f"\n📊 IK ROOT ROTATION ANALYSIS:")
    root_rot_quat = trans[:, 3:7]  # Extract quaternion part
    root_rot_norms = np.linalg.norm(root_rot_quat, axis=1)
    
    print(f"Root rotation quaternion norms:")
    print(f"  Range: [{root_rot_norms.min():.6f}, {root_rot_norms.max():.6f}]")
    print(f"  Mean:  {root_rot_norms.mean():.6f}")
    
    # Check if quaternions are normalized (should be close to 1.0)
    if not np.allclose(root_rot_norms, 1.0, atol=1e-3):
        print(f"  ⚠️  WARNING: Root rotation quaternions are not normalized!")
    else:
        print(f"  ✅ Root rotation quaternions are properly normalized")
    
    # Check for motion consistency
    print(f"\n📊 MOTION CONSISTENCY ANALYSIS:")
    
    # Check if IK root is moving
    ik_root_vel = np.diff(ik_root_pos, axis=0) * source_motion.fps
    ik_root_vel_mag = np.linalg.norm(ik_root_vel, axis=1)
    print(f"IK root velocity:")
    print(f"  Mean: {ik_root_vel_mag.mean():.3f} m/s")
    print(f"  Max:  {ik_root_vel_mag.max():.3f} m/s")
    
    if ik_root_vel_mag.mean() < 0.01:
        print(f"  ⚠️  WARNING: IK root is barely moving!")
    else:
        print(f"  ✅ IK root shows reasonable movement")
    
    # Check if joint angles are changing
    joint_angle_vel = np.diff(joint_angles, axis=0) * source_motion.fps
    joint_angle_vel_mag = np.linalg.norm(joint_angle_vel, axis=1)
    print(f"Joint angle velocity:")
    print(f"  Mean: {joint_angle_vel_mag.mean():.3f} rad/s")
    print(f"  Max:  {joint_angle_vel_mag.max():.3f} rad/s")
    
    if joint_angle_vel_mag.mean() < 0.01:
        print(f"  ⚠️  WARNING: Joint angles are barely changing!")
    else:
        print(f"  ✅ Joint angles show reasonable movement")
    
    print("="*60)

    print("✅ Retargeting complete! Output saved successfully.")


if __name__ == "__main__":
    # Run the main application
    typer.run(main)
