import typer
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Optional
import torch
import numpy as np
from dm_control import mjcf
from dm_control.viewer import user_input
from collections import OrderedDict

import mujoco
import mujoco.viewer

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState

import pandas as pd
import poselib.core.rotation3d as pRot
# tensor_to_dict not needed since we use numpy arrays directly
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from .treadmill2overground import transform_treadmill_to_overground


# --- Configuration for Your Lower Body Robot ---
_SMPL_HUMANOID_LOWER_BODY_KEYPOINT_TO_JOINT = {
    "Pelvis": {"name": "Pelvis", "weight": 1.0},
    "L_Hip": {"name": "L_Hip", "weight": 1.5},
    "R_Hip": {"name": "R_Hip", "weight": 1.5},
    "L_Knee": {"name": "L_Knee", "weight": 2.5},
    "R_Knee": {"name": "R_Knee", "weight": 2.0},  # Reduced weight for right knee
    "L_Ankle": {"name": "L_Ankle", "weight": 8.0},
    "R_Ankle": {"name": "R_Ankle", "weight": 5.0},  # Reduced weight for right ankle
    "L_Toe": {"name": "L_Toe", "weight": 10.0},
    "R_Toe": {"name": "R_Toe", "weight": 7.0},     # Reduced weight for right toe
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
    """A simple container to hold motion data for all joints."""
    global_translation: torch.Tensor  # All joint positions (including pelvis)
    global_rotation: torch.Tensor     # All joint rotations (local format)
    local_rotation: torch.Tensor      # All joint rotations (local format)
    skeleton_tree: SkeletonTree       # Complete skeleton tree
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

# --- Data Loading and Main Execution ---

def create_motion_from_txt(motion_filepath: str, rotation_filepath: str, tpose_filepath: str, tpose_rotation_filepath: str, treadmill_speed: float, mocap_fr: int = 200, debug_file: Optional[Path] = None, coordinate_transform: str = "y_to_x_forward"):
    """
    Loads motion data from position and quaternion files from biomechanics software.
    
    IMPORTANT: Quaternion format expectations:
    - All quaternions in WXYZ format (scalar-first: [w, x, y, z])
    - Pelvis quaternion: Global rotation (pelvis → lab coordinate system)
    - Other joint quaternions: Local rotations (parent bone → child bone)
    
    Args:
        motion_filepath: Path to .txt file with joint positions[x, y, z] format. txt is without headers but has frame numbers in first column. Joint order: Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, R_Knee, R_Ankle, R_Toe.
        rotation_filepath: Path to .txt file with joint quaternions [w, x, y, z] format (WXYZ - scalar first). Pelvis=global rotation, others=local rotations (parent→child). txt is without headers but has frame numbers in first column. Joint order: Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, R_Knee, R_Ankle, R_Toe.
        tpose_filepath: Path to T-pose .txt file with positions[x, y, z] format. txt is without headers but has frame numbers in first column. Joint order: Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, R_Knee, R_Ankle, R_Toe.
        tpose_rotation_filepath: Path to T-pose .txt file with quaternions [w, x, y, z] format (WXYZ - scalar first). txt is without headers but has frame numbers in first column. Joint order: Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, R_Knee, R_Ankle, R_Toe.
        treadmill_speed: Speed of treadmill for transformation
        mocap_fr: Frame rate of motion capture data
        debug_file: Optional debug file path
        coordinate_transform: Coordinate system transformation to apply
            - "none": No transformation (X=right, Y=forward, Z=up)
            - "y_to_x_forward": Rotate 90° around Z to change Y=forward to X=forward (default)
        
    Returns:
        SimpleMotion: Motion object with positions and local_rotation quaternions ready for retargeting
    """
    print(f"Loading motion data from {motion_filepath}...")
    
    # Define joint names based on the order specified in the docstring
    # Joint order: Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, R_Knee, R_Ankle, R_Toe
    joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe']
    print(f"Using predefined joint names: {joint_names}")

    motion_df = pd.read_csv(motion_filepath, sep='\t', header=None)
    motion_raw_data = motion_df.iloc[:, 1:].to_numpy()  # Skip first column (frame numbers)
    joint_centers = motion_raw_data.reshape((motion_raw_data.shape[0], -1, 3))

    print(f"Loading T-pose position data from {tpose_filepath}...")
    tpose_df = pd.read_csv(tpose_filepath, sep='\t', header=None)
    tpose_positions = tpose_df.iloc[0, 1:].to_numpy().reshape(-1, 3)  # Skip first column (frame numbers)
    
    print(f"Loading T-pose quaternion data from {tpose_rotation_filepath}...")
    tpose_rotation_df = pd.read_csv(tpose_rotation_filepath, sep='\t', header=None)
    tpose_rotation_raw = tpose_rotation_df.iloc[0, 1:].to_numpy()  # Skip first column (frame numbers)
    tpose_quaternions = tpose_rotation_raw.reshape(-1, 4)  # [w, x, y, z] format
    print(f"Loaded T-pose quaternions for {tpose_quaternions.shape[0]} joints")
    
    # Validate quaternion format - WXYZ quaternions should have reasonable magnitudes
    print("Validating T-pose quaternion format (expecting WXYZ [w, x, y, z])...")
    for i in range(min(3, tpose_quaternions.shape[0])):
        quat = tpose_quaternions[i]
        magnitude = np.linalg.norm(quat)
        print(f"  {joint_names[i]}: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}], mag: {magnitude:.3f}")
        
        if abs(magnitude - 1.0) > 0.1:
            print(f"  ⚠️  Warning: {joint_names[i]} quaternion magnitude {magnitude:.3f} is not close to 1.0")
            print("      This might indicate incorrect quaternion format or data corruption")

    # Apply coordinate system transformation if requested
    if coordinate_transform == "y_to_x_forward":
        print("Applying coordinate transformation: Y=forward → X=forward (rotating -90° around Z-axis)")
        
        # Rotate positions: [x, y, z] → [y, -x, z] (90° counter-clockwise around Z)
        joint_centers_transformed = joint_centers.copy()
        joint_centers_transformed[:, :, 0] = joint_centers[:, :, 1]   # new X = old Y
        joint_centers_transformed[:, :, 1] = -joint_centers[:, :, 0]  # new Y = -old X
        joint_centers = joint_centers_transformed
        
        # Transform T-pose positions the same way
        tpose_positions_transformed = tpose_positions.copy()
        tpose_positions_transformed[:, 0] = tpose_positions[:, 1]   # new X = old Y
        tpose_positions_transformed[:, 1] = -tpose_positions[:, 0]  # new Y = -old X
        tpose_positions = tpose_positions_transformed
        
        print("  ✅ Coordinate transformation applied to positions")
    elif coordinate_transform == "none":
        print("No coordinate transformation applied (keeping X=right, Y=forward, Z=up)")
    else:
        print(f"Warning: Unknown coordinate transform '{coordinate_transform}', skipping transformation")

    # Apply ground plane adjustment to ensure feet contact the ground
    print("Applying ground plane adjustment...")
    
    # Find the lowest point across all frames and joints (should be foot contact points)
    min_z_across_motion = np.min(joint_centers[:, :, 2])
    
    # Adjust all joints so the lowest point touches z=0
    ground_offset = -min_z_across_motion
    joint_centers[:, :, 2] += ground_offset
    tpose_positions[:, 2] += ground_offset
    
    print(f"  ✅ Adjusted motion by {ground_offset:.3f}m vertically to place feet on ground plane")
    print(f"  Motion Z-range: [{np.min(joint_centers[:, :, 2]):.3f}, {np.max(joint_centers[:, :, 2]):.3f}]m")

    # Apply treadmill-to-overground transformation if a non-zero treadmill speed is given.
    if treadmill_speed > 0:
        # Determine forward axis based on coordinate transformation
        forward_axis = 0 if coordinate_transform == "y_to_x_forward" else 1  # X=0 after transformation, Y=1 before
        joint_centers = transform_treadmill_to_overground(joint_centers, joint_names, mocap_fr, treadmill_speed, forward_axis)

    if debug_file:
        # save a copy of the processed joint centers for debugging with a name that indicates it's post-transformation
        debug_file = debug_file.with_name(debug_file.stem + "_treadmill_transform.npz")
        np.savez(debug_file, body_positions=joint_centers, body_names=joint_names, fps=mocap_fr)
        print(f"Saved debug data to: {debug_file}")

    # Create skeleton tree WITH all 9 joints (including pelvis for retargeting compatibility)
    # Keep original joint structure to match retargeting expectations
    parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7]
    
    # Use T-pose positions for proper bone lengths
    local_translations = np.zeros_like(tpose_positions)
    for i, p_idx in enumerate(parents):
        if p_idx == -1:
            local_translations[i] = tpose_positions[i]
        else:
            local_translations[i] = tpose_positions[i] - tpose_positions[p_idx]
            
    skeleton_tree = SkeletonTree(joint_names, torch.tensor(parents), torch.from_numpy(local_translations).float())
    print(f"Created skeleton tree with all joints: {len(joint_names)} joints")

    # Load quaternion data from biomechanics software
    print(f"Loading quaternion data from {rotation_filepath}...")
    rotation_df = pd.read_csv(rotation_filepath, sep='\t', header=None)
    rotation_raw_data = rotation_df.iloc[:, 1:].to_numpy()  # Skip first column (frame numbers)
    
    # Validate motion quaternion format
    print("Validating motion quaternion format (expecting WXYZ [w, x, y, z])...")
    motion_quaternions_sample = rotation_raw_data.reshape((rotation_raw_data.shape[0], -1, 4))
    for i in range(min(3, motion_quaternions_sample.shape[1])):
        # Check first frame quaternion for each joint
        quat = motion_quaternions_sample[0, i]
        magnitude = np.linalg.norm(quat)
        print(f"  {joint_names[i]} (frame 0): [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}], mag: {magnitude:.3f}")
        
        if abs(magnitude - 1.0) > 0.1:
            print(f"  ⚠️  Warning: {joint_names[i]} frame 0 quaternion magnitude {magnitude:.3f} is not close to 1.0")
            print("      This might indicate incorrect quaternion format or data corruption")
    
    # Process quaternion data 
    # Your data format: Pelvis = global rotation, Others = local rotations (parent→child)
    # Convert to ALL local rotations for poselib compatibility
    motion_quaternions = rotation_raw_data.reshape((rotation_raw_data.shape[0], -1, 4))
    
    # Convert pelvis from global to local (relative to T-pose)
    print("Converting pelvis from global to local rotation...")
    pelvis_global = motion_quaternions[:, 0, :]  # Pelvis quaternions (global)
    pelvis_tpose = tpose_quaternions[0]  # Pelvis T-pose (global)
    
    # Convert to local: local = global * tpose_inverse
    for frame in range(motion_quaternions.shape[0]):
        global_quat = pelvis_global[frame]
        # Convert [w,x,y,z] to scipy format [x,y,z,w]
        global_scipy = [global_quat[1], global_quat[2], global_quat[3], global_quat[0]]
        tpose_scipy = [pelvis_tpose[1], pelvis_tpose[2], pelvis_tpose[3], pelvis_tpose[0]]
        
        global_rot = R.from_quat(global_scipy)
        tpose_rot = R.from_quat(tpose_scipy)
        
        # Local = global * tpose^(-1)
        local_rot = global_rot * tpose_rot.inv()
        local_quat_scipy = local_rot.as_quat()  # [x,y,z,w]
        
        # Convert back to [w,x,y,z] format
        motion_quaternions[frame, 0] = [local_quat_scipy[3], local_quat_scipy[0], 
                                        local_quat_scipy[1], local_quat_scipy[2]]
    
    joint_rotations = torch.from_numpy(motion_quaternions).float()
    print(f"Converted to ALL local rotations: {joint_rotations.shape[0]} frames and {joint_rotations.shape[1]} joints")

    # Keep ALL joint data together (including pelvis) for retargeting compatibility
    print("Using all joint data (including pelvis) for proper retargeting...")
    
    # Convert all data to tensors for poselib compatibility
    all_joint_positions = torch.from_numpy(joint_centers).float()  # All 9 joints
    all_joint_rotations = joint_rotations  # Already tensor, all 9 joints
    
    print(f"All joints: {all_joint_positions.shape[1]} joints x {all_joint_positions.shape[0]} frames")
    print(f"Joint names: {joint_names}")
    
    # Create motion object with all joint data (retargeting will handle pelvis as root internally)
    motion = SimpleMotion(
        global_translation=all_joint_positions,  # All joint positions (9 joints)
        global_rotation=all_joint_rotations,     # All joint rotations (9 joints, local format)
        local_rotation=all_joint_rotations,      # Same as global_rotation (local format)
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
    rotation_file: Path = typer.Argument(..., help="Path to your .txt quaternion file [w, x, y, z] format."),
    tpose_file: Path = typer.Argument(..., help="Path to your T-pose .txt file."),
    tpose_rotation_file: Path = typer.Argument(..., help="Path to your T-pose quaternion .txt file [w, x, y, z] format."),
    output_file: Path = typer.Argument(..., help="Path to save the output .npy file."),
    robot_xml: Path = typer.Argument(..., help="Path to your robot's .xml file."),
    treadmill_speed: float = typer.Option(3.0, help="Treadmill speed in m/s."),
    coordinate_transform: str = typer.Option("y_to_x_forward", help="Coordinate system transformation: 'none' (X=right, Y=forward) or 'y_to_x_forward' (Y=forward → X=forward)"),
    render: bool = typer.Option(False, help="Enable live rendering of the retargeting process."),
    debug_file: Optional[Path] = typer.Option(None, help="Path to save the debug .npz file."),
):
    """
    This script retargets motion from text files of 3D joint positions and quaternions to a
    lower-body humanoid robot for use in Isaac Lab.
    
    Both motion and T-pose rotation files should contain quaternions in [w, x, y, z] format (scalar-first).
    The script computes relative rotations from T-pose to each motion frame, providing
    accurate joint angles relative to the anatomical reference configuration.
    
    Coordinate System Options:
    - 'none': Keep original coordinate system (X=right, Y=forward, Z=up)
    - 'y_to_x_forward': Transform to robotics convention (X=forward, Y=left, Z=up)
    
    This approach uses quaternion data directly from biomechanics software with
    T-pose reference for the most accurate joint orientations.
    """
    robot_type = "smpl_humanoid_lower_body"

    source_motion = create_motion_from_txt(str(motion_file), str(rotation_file), str(tpose_file), str(tpose_rotation_file), treadmill_speed, debug_file=debug_file, coordinate_transform=coordinate_transform)

    # Retarget the motion to the target robot using Mink
    print(f"Retargeting motion to {robot_type} using Mink...")
    retargeted_motion = retarget_motion_to_robot(source_motion, robot_type, render=render)
    
    # Save debug NPZ file after retargeting for visualization
    if debug_file:
        retarget_debug_file = debug_file.with_name(debug_file.stem + "_retargeted")
        retargeted_positions = retargeted_motion.global_translation.numpy()
        retargeted_joint_names = retargeted_motion.skeleton_tree.node_names
        np.savez(
            retarget_debug_file, 
            body_positions=retargeted_positions, 
            body_names=retargeted_joint_names, 
            fps=retargeted_motion.fps
        )
        print(f"Saved retargeted debug data to: {retarget_debug_file}")
        print(f"You can visualize this with: python data/scripts/debug_motion_viewer.py --file {retarget_debug_file}")
    
    # Save the retargeted motion with numpy arrays (not dicts) for np.savez compatibility
    print(f"Saving retargeted motion to {output_file}")
    output_data = OrderedDict([
        ("rotation", retargeted_motion.global_rotation.numpy()),
        ("root_translation", retargeted_motion.root_translation.numpy()),  
        ("global_velocity", retargeted_motion.global_velocity.numpy()),
        ("global_angular_velocity", retargeted_motion.global_angular_velocity.numpy()),
        # Note: Skeleton tree saved separately due to its complex structure
        ("fps", np.array(retargeted_motion.fps, dtype=np.float64)),
        ("is_local", np.array(False)),
    ])

    print(f"Saving {retargeted_motion.global_rotation.shape[0]} frames of retargeted motion to: {output_file}")
    np.savez(output_file, **output_data)



if __name__ == "__main__":
    # Run the main application
    typer.run(main)
