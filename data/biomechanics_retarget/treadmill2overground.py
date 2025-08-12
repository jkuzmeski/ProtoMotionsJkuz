# -*- coding: utf-8 -*-
"""
Treadmill to Overground Motion Transformation Script

This script processes lower-body motion capture data from a treadmill experiment,
transforms it to appear as if performed overground, and saves the results in
multiple formats (.npy, .txt, .csv, .json).

Key Processing Steps:
1.  Loads joint position data from a text file.
2.  Applies an optional coordinate system transformation (e.g., Y-forward to X-forward).
3.  Adjusts the motion vertically to ensure the feet contact the ground plane (z=0).
4.  Transforms treadmill motion to overground motion by:
    a. Detecting foot stance phases using kinematic criteria (position, velocity, acceleration).
    b. Stabilizing the stance foot/feet in each frame to remove backward drift.
    c. Applying a consistent forward velocity to the entire body based on the
       treadmill speed to create realistic overground progression.
5.  Saves the transformed joint positions and associated metadata.
"""
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import typer
from scipy.ndimage import binary_closing, binary_opening
from scipy.spatial.transform import Rotation as R

# --- Constants ---
# This order must match the joint order in the input text files.
JOINT_NAMES = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
]

# --- Core Functions ---


def parse_speed_from_filename(filename: str) -> Optional[float]:
    """
    Extracts treadmill speed from filename.
    
    Expected format: S##_##ms_positions_lowerbody.txt
    Where ##ms represents speed * 10 (e.g., 20ms = 2.0 m/s, 45ms = 4.5 m/s)
    
    Args:
        filename (str): The filename to parse.
        
    Returns:
        Optional[float]: The parsed speed in m/s, or None if parsing fails.
    """
    # Pattern to match S##_##ms_positions_lowerbody.txt
    pattern = r'S\d+_(\d+)ms_.*\.txt'
    match = re.search(pattern, filename)
    
    if match:
        speed_ms = int(match.group(1))
        speed_mps = speed_ms / 10.0  # Convert from ##ms to m/s
        return speed_mps
    
    return None


def process_motion_file(
    motion_file: Path,
    output_dir: Path,
    fps: int,
    coordinate_transform: str,
    speed_override: Optional[float] = None,
) -> bool:
    """
    Processes a single motion file.
    
    Args:
        motion_file (Path): Path to the motion file.
        output_dir (Path): Directory to save output files.
        fps (int): Frame rate.
        coordinate_transform (str): Coordinate transformation to apply.
        speed_override (Optional[float]): Override speed instead of parsing from filename.
        
    Returns:
        bool: True if processing succeeded, False otherwise.
    """
    print(f"\n📁 Processing file: {motion_file.name}")
    
    # Parse speed from filename or use override
    if speed_override is not None:
        treadmill_speed = speed_override
        print(f"   - Using override speed: {treadmill_speed:.1f} m/s")
    else:
        treadmill_speed = parse_speed_from_filename(motion_file.name)
        if treadmill_speed is None:
            print(f"   - ⚠️ Could not parse speed from filename '{motion_file.name}'. Skipping.")
            return False
        print(f"   - Parsed speed from filename: {treadmill_speed:.1f} m/s")
    
    try:
        # Load, process, and transform motion data
        joint_centers = create_motion_from_txt(
            motion_filepath=str(motion_file),
            treadmill_speed=treadmill_speed,
            mocap_fr=fps,
            coordinate_transform=coordinate_transform,
        )
        
        # Create output filename based on input filename (remove extension)
        output_base = output_dir / motion_file.stem
        
        # Save the transformed motion data to multiple formats
        save_motion_data(
            joint_centers=joint_centers,
            output_path_base=str(output_base),
            fps=fps,
            transform_applied=coordinate_transform,
        )
        
        print(f"   - ✅ Successfully processed {motion_file.name}")
        return True
        
    except Exception as e:
        print(f"   - ❌ Error processing {motion_file.name}: {e}")
        return False


def calculate_kinematics(
    positions: np.ndarray, fps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates velocities and accelerations from position data using np.gradient.

    This method is robust and handles boundary conditions gracefully by using
    second-order accurate finite differences.

    Args:
        positions (np.ndarray): Array of position data (n_frames, ...).
        fps (int): The frame rate of the data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - velocities (np.ndarray): Calculated velocities in m/s.
            - accelerations (np.ndarray): Calculated accelerations in m/s^2.
    """
    if positions.shape[0] < 2:
        return np.zeros_like(positions), np.zeros_like(positions)

    time_delta = 1.0 / fps
    velocities = np.gradient(positions, time_delta, axis=0)
    accelerations = np.gradient(velocities, time_delta, axis=0)
    return velocities, accelerations


def detect_stance_phases(
    foot_positions: np.ndarray,
    foot_velocities: np.ndarray,
    foot_accelerations: np.ndarray,
    height_threshold: float = 0.05,
    vertical_velocity_threshold: float = 0.1,
    horizontal_acceleration_threshold: float = 0.5,
) -> np.ndarray:
    """
    Detects stance phases based on robust biomechanical criteria.

    A foot is in a stance phase if it is:
    1. Close to the ground (low height).
    2. Has minimal vertical movement (low vertical velocity).
    3. Moving at a constant horizontal velocity (low horizontal acceleration).

    Args:
        foot_positions (np.ndarray): Foot position data (n_frames, 3).
        foot_velocities (np.ndarray): Foot velocity data (n_frames, 3).
        foot_accelerations (np.ndarray): Foot acceleration data (n_frames, 3).
        height_threshold (float): Max height for ground contact in meters.
        vertical_velocity_threshold (float): Max vertical velocity for being stationary.
        horizontal_acceleration_threshold (float): Max horizontal acceleration for constant velocity.

    Returns:
        np.ndarray: A boolean array where True indicates a stance phase.
    """
    # Condition 1: Foot is close to the ground (Z-axis).
    height_condition = foot_positions[:, 2] < height_threshold

    # Condition 2: Foot has minimal vertical velocity.
    vertical_velocity_condition = (
        np.abs(foot_velocities[:, 2]) < vertical_velocity_threshold
    )

    # Condition 3: Foot has minimal horizontal acceleration (XY plane).
    horizontal_acceleration = np.linalg.norm(foot_accelerations[:, :2], axis=1)
    horizontal_acceleration_condition = (
        horizontal_acceleration < horizontal_acceleration_threshold
    )

    # Combine all conditions. A foot is in stance if all are true.
    stance_mask = (
        height_condition
        & vertical_velocity_condition
        & horizontal_acceleration_condition
    )

    # Clean up short, isolated stance/swing detections (noise) using morphological operations.
    stance_mask = binary_closing(stance_mask, structure=np.ones(5))
    stance_mask = binary_opening(stance_mask, structure=np.ones(3))

    return stance_mask


def transform_treadmill_to_overground(
    joint_centers: np.ndarray, fps: int, treadmill_speed: float, forward_axis: int
) -> np.ndarray:
    """
    Transforms treadmill running motion to appear as if performed overground.

    The process involves two main stages:
    1.  **Foot Stabilization**: The motion is adjusted frame-by-frame to ensure the
        stance foot remains stationary relative to the world, canceling out the
        backward movement caused by the treadmill belt.
    2.  **Forward Progression**: A linear forward displacement, corresponding to the
        treadmill speed, is added to the entire body to create a continuous
        overground movement.

    Args:
        joint_centers (np.ndarray): Joint positions (n_frames, n_joints, 3).
        fps (int): Frames per second of the motion data.
        treadmill_speed (float): Treadmill speed in m/s.
        forward_axis (int): The index of the forward-facing axis (0=X, 1=Y).

    Returns:
        np.ndarray: Transformed joint positions (n_frames, n_joints, 3).
    """
    print("🚀 Applying data-driven treadmill-to-overground transformation...")

    try:
        # Prefer ankle, but fall back to toe for foot tracking.
        l_foot_idx = (
            JOINT_NAMES.index("L_Ankle")
            if "L_Ankle" in JOINT_NAMES
            else JOINT_NAMES.index("L_Toe")
        )
        r_foot_idx = (
            JOINT_NAMES.index("R_Ankle")
            if "R_Ankle" in JOINT_NAMES
            else JOINT_NAMES.index("R_Toe")
        )
        pelvis_idx = JOINT_NAMES.index("Pelvis")
    except ValueError as e:
        print(f"⚠️ Warning: A required joint was not found ({e}). Skipping transformation.")
        return joint_centers.copy()

    n_frames = joint_centers.shape[0]
    l_foot_pos = joint_centers[:, l_foot_idx, :]
    r_foot_pos = joint_centers[:, r_foot_idx, :]

    # Calculate foot kinematics and detect stance phases.
    l_foot_vel, l_foot_accel = calculate_kinematics(l_foot_pos, fps)
    r_foot_vel, r_foot_accel = calculate_kinematics(r_foot_pos, fps)
    l_stance = detect_stance_phases(l_foot_pos, l_foot_vel, l_foot_accel)
    r_stance = detect_stance_phases(r_foot_pos, r_foot_vel, r_foot_accel)

    # --- Stage 1: Stabilize Stance Feet ---
    transformed_centers = joint_centers.copy()
    current_offset = np.zeros(3)
    last_offset_update = np.zeros(3)

    for i in range(1, n_frames):
        in_double_stance = l_stance[i] and r_stance[i]
        in_left_stance = l_stance[i] and not r_stance[i]
        in_right_stance = r_stance[i] and not l_stance[i]
        in_flight = not l_stance[i] and not r_stance[i]

        offset_update = np.zeros(3)
        if in_left_stance:
            # Cancel out left foot displacement.
            offset_update = -(l_foot_pos[i] - l_foot_pos[i - 1])
        elif in_right_stance:
            # Cancel out right foot displacement.
            offset_update = -(r_foot_pos[i] - r_foot_pos[i - 1])
        elif in_double_stance:
            # Use the average of both feet for smooth transitions.
            avg_foot_disp = ((l_foot_pos[i] + r_foot_pos[i]) / 2.0) - (
                (l_foot_pos[i - 1] + r_foot_pos[i - 1]) / 2.0
            )
            offset_update = -avg_foot_disp
        elif in_flight:
            # During flight, maintain momentum by reapplying the last ground-contact displacement.
            offset_update = last_offset_update

        # The transformation should only affect horizontal movement (XY plane).
        # Vertical (Z) motion is preserved from the original capture.
        offset_update[2] = 0.0
        current_offset += offset_update
        transformed_centers[i, :, :] += current_offset

        if not in_flight:
            last_offset_update = offset_update

    # --- Stage 2: Apply Forward Progression ---
    # Add a linear displacement to ensure the average speed matches the treadmill.
    time_vector = np.linspace(0, (n_frames - 1) / fps, n_frames)
    forward_displacement = treadmill_speed * time_vector
    transformed_centers[:, :, forward_axis] += forward_displacement[:, np.newaxis]

    # --- Final Verification ---
    duration = (n_frames - 1) / fps
    initial_pelvis_pos = joint_centers[0, pelvis_idx, forward_axis]
    final_pelvis_pos = transformed_centers[-1, pelvis_idx, forward_axis]
    total_forward_dist = final_pelvis_pos - initial_pelvis_pos
    axis_name = ["X", "Y", "Z"][forward_axis]

    print("✅ Transformation applied.")
    print(f"   - Pelvis moved {total_forward_dist:.2f}m forward in the {axis_name} direction.")
    print(
        f"   - Expected distance for {treadmill_speed:.1f} m/s over {duration:.1f}s: {treadmill_speed * duration:.2f}m"
    )
    return transformed_centers


def create_motion_from_txt(
    motion_filepath: str,
    treadmill_speed: float,
    mocap_fr: int,
    coordinate_transform: str,
) -> np.ndarray:
    """
    Loads, processes, and transforms motion data from a text file.
    """
    print(f"🔵 Loading motion data from {motion_filepath}...")
    try:
        motion_df = pd.read_csv(motion_filepath, sep="\t", header=None)
        # Skip first column (frame numbers), reshape to (frames, joints, 3)
        joint_centers = motion_df.iloc[:, 1:].to_numpy().reshape(motion_df.shape[0], -1, 3)
        print(f"   - Loaded {joint_centers.shape[0]} frames for {joint_centers.shape[1]} joints.")
    except Exception as e:
        print(f"❌ Error loading motion file {motion_filepath}: {e}")
        raise

    # Apply coordinate system transformation if requested.
    forward_axis = 1  # Default: Y-forward
    if coordinate_transform == "y_to_x_forward":
        print("🔵 Applying coordinate transformation: Y-forward -> X-forward.")
        # Create a rotation object for -90 degrees around the Z axis.
        rot = R.from_euler("z", -90, degrees=True)
        n_frames, n_joints, _ = joint_centers.shape
        # Apply rotation to all joint positions across all frames.
        joint_centers = rot.apply(joint_centers.reshape(-1, 3)).reshape(
            n_frames, n_joints, 3
        )
        forward_axis = 0  # New: X-forward

    # Apply ground plane adjustment to set the lowest foot contact point to z=0.
    print("🔵 Applying ground plane adjustment...")
    min_z = np.min(joint_centers[:, :, 2])
    joint_centers[:, :, 2] -= min_z
    print(f"   - Lowered motion by {-min_z:.3f}m to place on ground plane.")

    # Apply the main treadmill-to-overground transformation.
    if treadmill_speed > 0:
        joint_centers = transform_treadmill_to_overground(
            joint_centers, mocap_fr, treadmill_speed, forward_axis
        )
    else:
        print("🔵 Skipping treadmill transformation (speed is 0).")

    return joint_centers


def save_motion_data(
    joint_centers: np.ndarray,
    output_path_base: str,
    fps: int,
    transform_applied: str,
):
    """
    Saves transformed joint centers to .npy, .txt, .csv, and .json formats.
    The .npy file is saved to the main output path, while other files are saved to a metadata subfolder.
    """
    print(f"\n💾 Saving transformed motion data to '{output_path_base}.*'")
    output_path = Path(output_path_base)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create metadata subfolder
    metadata_dir = output_path.parent / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path_base = metadata_dir / output_path.name

    n_frames, n_joints, _ = joint_centers.shape

    # --- Save as NPY (binary format) to main output folder ---
    np.save(output_path.with_suffix(".npy"), joint_centers)

    # --- Save as TXT (tab-separated values) to metadata folder ---
    with open(metadata_path_base.with_suffix(".txt"), "w") as f:
        f.write("# Transformed Motion Data\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Joint Order: {', '.join(JOINT_NAMES)}\n")
        # Flatten each frame's data into a single row.
        flat_data = joint_centers.reshape(n_frames, -1)
        for i, row_data in enumerate(flat_data):
            # Prepend 1-based frame number.
            f.write(f"{i + 1}\t" + "\t".join(f"{x:.6f}" for x in row_data) + "\n")

    # --- Save as CSV (for spreadsheets) to metadata folder ---
    csv_columns = ["Frame"] + [
        f"{name}_{ax}" for name in JOINT_NAMES for ax in ["X", "Y", "Z"]
    ]
    # Add frame numbers as the first column.
    csv_data = np.hstack(
        (np.arange(1, n_frames + 1)[:, np.newaxis], joint_centers.reshape(n_frames, -1))
    )
    pd.DataFrame(csv_data, columns=csv_columns).to_csv(
        metadata_path_base.with_suffix(".csv"), index=False, float_format="%.6f"
    )

    # --- Save Metadata as JSON to metadata folder ---
    coord_system_desc = (
        "X=forward, Y=left, Z=up"
        if transform_applied == "y_to_x_forward"
        else "X=right, Y=forward, Z=up"
    )
    metadata = {
        "generated_timestamp": datetime.now().isoformat(),
        "fps": fps,
        "duration_seconds": n_frames / fps,
        "num_frames": n_frames,
        "num_joints": n_joints,
        "joint_names": JOINT_NAMES,
        "units": "meters",
        "coordinate_system": coord_system_desc,
    }
    with open(metadata_path_base.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    print("   - ✅ Saved .npy file to main output folder.")
    print("   - ✅ Saved .txt, .csv, and .json files to metadata folder.")


# --- Command-Line Interface ---
def main(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Path to motion file or folder containing motion files."
    ),
    output_path: str = typer.Argument(
        ..., help="Base output path for transformed motion (e.g., 'output/') or specific file path."
    ),
    treadmill_speed: Optional[float] = typer.Option(
        None, "--speed", "-s", help="Override treadmill speed in m/s. If not provided, speed will be parsed from filenames."
    ),
    fps: int = typer.Option(200, "--fps", "-f", help="Motion capture frame rate (Hz)."),
    coordinate_transform: str = typer.Option(
        "y_to_x_forward",
        "--transform",
        "-t",
        help="Coordinate transform: 'none' or 'y_to_x_forward'.",
    ),
):
    """
    Transforms treadmill motion capture data to overground motion.
    
    Can process either:
    - A single motion file (will parse speed from filename or use --speed)
    - A folder containing multiple motion files (will process all .txt files)
    """
    print("🏃 Treadmill-to-Overground Motion Transformer 🏃")
    print("=" * 50)
    
    try:
        output_dir = Path(output_path)
        
        if input_path.is_file():
            # Single file processing
            print(f"📄 Processing single file: {input_path}")
            
            if input_path.suffix.lower() != '.txt':
                print("❌ Input file must be a .txt file.")
                raise typer.Exit(code=1)
            
            # For single file, output_path can be specific file or directory
            if output_dir.suffix:
                # Specific output file specified
                output_base = str(output_dir.with_suffix(''))
                output_dir.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Output directory specified
                output_dir.mkdir(parents=True, exist_ok=True)
                output_base = str(output_dir / input_path.stem)
            
            success = process_motion_file(
                motion_file=input_path,
                output_dir=output_dir.parent if output_dir.suffix else output_dir,
                fps=fps,
                coordinate_transform=coordinate_transform,
                speed_override=treadmill_speed,
            )
            
            if success:
                print("\n🎉 Processing completed successfully!")
                print(f"   - Output files created in: {Path(output_base).parent}")
            else:
                print("\n❌ Processing failed.")
                raise typer.Exit(code=1)
                
        elif input_path.is_dir():
            # Folder processing
            print(f"📁 Processing folder: {input_path}")
            
            # Find all .txt files in the directory
            motion_files = list(input_path.glob("*.txt"))
            
            if not motion_files:
                print("❌ No .txt files found in the specified directory.")
                raise typer.Exit(code=1)
            
            print(f"   - Found {len(motion_files)} .txt files to process")
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each file
            successful_files = 0
            failed_files = 0
            
            for motion_file in sorted(motion_files):
                success = process_motion_file(
                    motion_file=motion_file,
                    output_dir=output_dir,
                    fps=fps,
                    coordinate_transform=coordinate_transform,
                    speed_override=treadmill_speed,
                )
                
                if success:
                    successful_files += 1
                else:
                    failed_files += 1
            
            print("\n🎉 Batch processing completed!")
            print(f"   - Successfully processed: {successful_files} files")
            print(f"   - Failed to process: {failed_files} files")
            print(f"   - Output files created in: {output_dir}")
            
            if failed_files > 0:
                print("   - ⚠️ Some files failed to process. Check the logs above for details.")
        
        else:
            print("❌ Input path must be either a file or directory.")
            raise typer.Exit(code=1)

    except Exception as e:
        print(f"\n❌ An error occurred during processing: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main) 
