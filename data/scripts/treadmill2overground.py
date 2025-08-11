import typer
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import json
from datetime import datetime

from poselib.skeleton.skeleton3d import SkeletonTree

import pandas as pd


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
        # This class is kept for potential future use but KEY_SPACE is not used currently
        self.pause = not self.pause


@dataclass
class SimpleMotion:
    """A simple container to hold motion data for all joints."""
    global_translation: np.ndarray    # All joint positions (including pelvis)
    global_rotation: np.ndarray       # All joint rotations (global format)
    local_rotation: np.ndarray        # All joint rotations (local format)
    skeleton_tree: SkeletonTree       # Complete skeleton tree
    fps: int


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


def transform_treadmill_to_overground(joint_centers, joint_names, fps, treadmill_speed, forward_axis=1):
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
        forward_axis (int): Axis index for forward direction (0=X, 1=Y, 2=Z). Default 1 for Y-forward.

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
        
        # The transformation should only affect horizontal movement.
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
        transformed_centers[frame, :, forward_axis] += additional_forward_offset
    
    # Final verification
    final_pelvis_pos = transformed_centers[-1, pelvis_idx, :]
    total_forward_dist = final_pelvis_pos[forward_axis] - initial_pelvis_pos[forward_axis]
    
    axis_names = ['X', 'Y', 'Z']
    print(f"Applied transformation: pelvis moved {total_forward_dist:.3f}m forward in {axis_names[forward_axis]} direction.")
    print(f"Expected distance for {treadmill_speed:.1f} m/s over {duration:.1f}s: {expected_forward_distance:.3f}m")
    return transformed_centers


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
        np.ndarray: (n_frames, n_joints, 3) array of joint centers ready for saving
    """
    print(f"Loading motion data from {motion_filepath}...")
    
    # Define joint names based on the order specified in the docstring
    # Joint order: Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, R_Knee, R_Ankle, R_Toe
    joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe']
    print(f"Using predefined joint names: {joint_names}")

    # Load motion position data
    try:
        motion_df = pd.read_csv(motion_filepath, sep='\t', header=None)
        motion_raw_data = motion_df.iloc[:, 1:].to_numpy()  # Skip first column (frame numbers)
        joint_centers = motion_raw_data.reshape((motion_raw_data.shape[0], -1, 3))
        print(f"✅ Loaded {joint_centers.shape[0]} frames of motion data")
    except Exception as e:
        print(f"❌ Error loading motion file {motion_filepath}: {e}")
        raise

    # Load T-pose data (currently loaded but not used - kept for future enhancements)
    try:
        print(f"Loading T-pose position data from {tpose_filepath}...")
        tpose_df = pd.read_csv(tpose_filepath, sep='\t', header=None)
        tpose_positions = tpose_df.iloc[0, 1:].to_numpy().reshape(-1, 3)  # Skip first column (frame numbers)
        
        print(f"Loading T-pose quaternion data from {tpose_rotation_filepath}...")
        tpose_rotation_df = pd.read_csv(tpose_rotation_filepath, sep='\t', header=None)
        tpose_rotation_raw = tpose_rotation_df.iloc[0, 1:].to_numpy()  # Skip first column (frame numbers)
        tpose_quaternions = tpose_rotation_raw.reshape(-1, 4)  # [w, x, y, z] format
        print(f"✅ Loaded T-pose data for {tpose_quaternions.shape[0]} joints")
    except Exception as e:
        print(f"⚠️  Warning: Could not load T-pose data: {e}")
        print("   Proceeding without T-pose validation...")
        tpose_positions = None
        tpose_quaternions = None
    
    # Validate quaternion format if T-pose data is available
    if tpose_quaternions is not None:
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
        
        # Transform T-pose positions the same way if available
        if tpose_positions is not None:
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
    if tpose_positions is not None:
        tpose_positions[:, 2] += ground_offset
    
    print(f"  ✅ Adjusted motion by {ground_offset:.3f}m vertically to place feet on ground plane")
    print(f"  Motion Z-range: [{np.min(joint_centers[:, :, 2]):.3f}, {np.max(joint_centers[:, :, 2]):.3f}]m")

    # Apply treadmill-to-overground transformation if a non-zero treadmill speed is given.
    if treadmill_speed > 0:
        print(f"Applying treadmill-to-overground transformation (speed: {treadmill_speed} m/s)...")
        # Determine forward axis based on coordinate transformation
        forward_axis = 0 if coordinate_transform == "y_to_x_forward" else 1  # X=0 after transformation, Y=1 before
        joint_centers = transform_treadmill_to_overground(joint_centers, joint_names, mocap_fr, treadmill_speed, forward_axis)
    else:
        print("Skipping treadmill transformation (speed = 0)")

    return joint_centers


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


def save_joint_centers(joint_centers, joint_names, output_path, fps=200, include_metadata=True):
    """
    Save transformed joint centers to various formats.
    
    Args:
        joint_centers (np.ndarray): (n_frames, n_joints, 3) array of joint positions
        joint_names (list): List of joint names
        output_path (str): Base output path (without extension)
        fps (int): Frame rate for metadata
        include_metadata (bool): Whether to include metadata in saved files
    """
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_frames, n_joints, _ = joint_centers.shape
    
    # Save as numpy array (.npy)
    npy_path = output_path.with_suffix('.npy')
    np.save(npy_path, joint_centers)
    print(f"✅ Saved joint centers as numpy array: {npy_path}")
    
    # Save as tab-separated text file (similar to input format)
    txt_path = output_path.with_suffix('.txt')
    with open(txt_path, 'w') as f:
        # Optional header with joint names
        if include_metadata:
            f.write(f"# Joint order: {', '.join(joint_names)}\n")
            f.write(f"# Shape: {n_frames} frames, {n_joints} joints, 3 coordinates (x, y, z)\n")
            f.write(f"# FPS: {fps}\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Write data with frame numbers
        for frame_idx in range(n_frames):
            # Start with frame number (1-indexed to match input format)
            row = [str(frame_idx + 1)]
            
            # Add all joint coordinates
            for joint_idx in range(n_joints):
                for coord_idx in range(3):
                    row.append(f"{joint_centers[frame_idx, joint_idx, coord_idx]:.6f}")
            
            f.write('\t'.join(row) + '\n')
    
    print(f"✅ Saved joint centers as text file: {txt_path}")
    
    # Save metadata as JSON
    if include_metadata:
        metadata = {
            'joint_names': joint_names,
            'shape': list(joint_centers.shape),
            'fps': fps,
            'units': 'meters',
            'coordinate_system': 'X=right, Y=forward, Z=up (after transformation)',
            'generated_timestamp': datetime.now().isoformat(),
            'data_description': 'Transformed joint center positions (treadmill to overground)'
        }
        
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Saved metadata: {json_path}")
    
    # Save as CSV for easy viewing in spreadsheet applications
    csv_path = output_path.with_suffix('.csv')
    
    # Create column names
    columns = ['Frame']
    for joint_name in joint_names:
        columns.extend([f"{joint_name}_X", f"{joint_name}_Y", f"{joint_name}_Z"])
    
    # Reshape data for DataFrame
    data_for_df = []
    for frame_idx in range(n_frames):
        row = [frame_idx + 1]  # Frame number (1-indexed)
        for joint_idx in range(n_joints):
            row.extend(joint_centers[frame_idx, joint_idx, :])
        data_for_df.append(row)
    
    df = pd.DataFrame(data_for_df, columns=columns)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved joint centers as CSV: {csv_path}")
    
    print("\n📊 Summary:")
    print(f"   • {n_frames} frames, {n_joints} joints")
    print(f"   • Motion duration: {n_frames/fps:.2f} seconds at {fps} FPS")
    print(f"   • Files saved in: {output_dir}")
    
    return {
        'npy': npy_path,
        'txt': txt_path,
        'csv': csv_path,
        'json': json_path if include_metadata else None
    }


def main(
    motion_file: str = typer.Argument(..., help="Path to motion positions text file"),
    rotation_file: str = typer.Argument(..., help="Path to motion quaternions text file"),
    tpose_file: str = typer.Argument(..., help="Path to T-pose positions text file"),
    tpose_rotation_file: str = typer.Argument(..., help="Path to T-pose quaternions text file"),
    output_path: str = typer.Argument(..., help="Output path for transformed motion (without extension)"),
    treadmill_speed: float = typer.Option(1.5, "--speed", "-s", help="Treadmill speed in m/s"),
    fps: int = typer.Option(200, "--fps", "-f", help="Motion capture frame rate"),
    coordinate_transform: str = typer.Option(
        "y_to_x_forward", "--transform", "-t", 
        help="Coordinate transformation: 'none' or 'y_to_x_forward'"
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug output")
):
    """
    Transform treadmill motion data to overground motion and save the results.
    
    This script processes motion capture data from a treadmill experiment and transforms
    it to appear as if the motion was performed overground. The transformation stabilizes
    stance feet and adds forward progression based on treadmill speed.
    
    Example usage:
        python treadmill2overground.py motion.txt rotations.txt tpose.txt tpose_rot.txt output_motion --speed 2.0 --fps 200
    """
    print("🏃 Treadmill to Overground Motion Transformer")
    print("=" * 50)
    
    try:
        # Load and transform motion data
        joint_centers = create_motion_from_txt(
            motion_filepath=motion_file,
            rotation_filepath=rotation_file,
            tpose_filepath=tpose_file,
            tpose_rotation_filepath=tpose_rotation_file,
            treadmill_speed=treadmill_speed,
            mocap_fr=fps,
            coordinate_transform=coordinate_transform
        )
        
        # Define joint names (should match the order in your data files)
        joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe']
        
        # Save the transformed motion data
        print("\n💾 Saving transformed motion data...")
        save_joint_centers(
            joint_centers=joint_centers,
            joint_names=joint_names,
            output_path=output_path,
            fps=fps,
            include_metadata=True
        )
        
        print("\n✅ Processing completed successfully!")
        print(f"   Original motion: {joint_centers.shape[0]} frames")
        print(f"   Treadmill speed: {treadmill_speed} m/s")
        print(f"   Transform applied: {coordinate_transform}")
        
        if debug:
            print("\n🔍 Debug information:")
            print(f"   Joint centers shape: {joint_centers.shape}")
            print(f"   Joint names: {joint_names}")
            print(f"   Motion range X: [{np.min(joint_centers[:,:,0]):.3f}, {np.max(joint_centers[:,:,0]):.3f}]")
            print(f"   Motion range Y: [{np.min(joint_centers[:,:,1]):.3f}, {np.max(joint_centers[:,:,1]):.3f}]")
            print(f"   Motion range Z: [{np.min(joint_centers[:,:,2]):.3f}, {np.max(joint_centers[:,:,2]):.3f}]")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    # Run the main application
    typer.run(main)
