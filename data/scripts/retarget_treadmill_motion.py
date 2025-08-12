"""
Retarget motion data from treadmill2overground.py output to SMPL lower body model format.
This script takes the joint center positions output from treadmill2overground.py and
converts them to the SkeletonMotion format that's compatible with the ProtoMotions
data pipeline.
"""

import typer
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Sequence
from lxml import etree

from scipy.spatial.transform import Rotation as sRot
import matplotlib.pyplot as plt
import matplotlib.animation

import mpl_toolkits.mplot3d  # noqa: F401

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState

import mink
import mujoco
import mujoco.viewer
from dm_control import mjcf
from contextlib import nullcontext
from loop_rate_limiters import RateLimiter
from tqdm import tqdm

# ---- Mink Retargeting Constants ----

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

# ---- MuJoCo Model Helper Functions ----


def to_string(
    root: mjcf.RootElement,
    precision: int = 17,
    zero_threshold: float = 1e-7,
    pretty: bool = False,
) -> str:

    xml_string = root.to_xml_string(precision=precision, zero_threshold=zero_threshold)
    xml_root = etree.XML(xml_string, etree.XMLParser(remove_blank_text=True))

    if not pretty:
        return etree.tostring(xml_root, pretty_print=True).decode()

    # Remove auto-generated names.
    for elem in xml_root.iter():
        for key in list(elem.keys()):
            if key == "name" and "unnamed" in elem.get(key, ""):
                del elem.attrib[key]

    # Get string from lxml.
    xml_string = etree.tostring(xml_root, pretty_print=True)
    return xml_string.decode()


def construct_model(robot_name: str, keypoint_names: Sequence[str]):
    """Constructs a MuJoCo model for retargeting."""
    root = mjcf.RootElement()
    if root.visual:
        root.visual.headlight.ambient = np.array([0.4, 0.4, 0.4])
        root.visual.headlight.diffuse = np.array([0.8, 0.8, 0.8])
        root.visual.headlight.specular = np.array([0.1, 0.1, 0.1])

    # Add ground
    root.asset.add("texture", name="grid", type="2d", builtin="checker", rgb1=".1 .2 .3", rgb2=".2 .3 .4", width="300", height="300")
    root.asset.add("material", name="grid", texture="grid", texrepeat="1 1", texuniform="true", reflectance=".2")
    root.worldbody.add("geom", name="ground", type="plane", size="0 0 .01", material="grid")

    # Add mocap bodies for keypoints
    for keypoint_name in keypoint_names:
        body = root.worldbody.add("body", name=f"keypoint_{keypoint_name}", mocap="true")
        rgb = np.random.rand(3)
        body.add("site", name=f"site_{keypoint_name}", type="sphere", size="0.02", rgba=f"{rgb[0]} {rgb[1]} {rgb[2]} 1")

    # Load robot model
    xml_path = Path(__file__).parent.parent.parent / "protomotions" / "data" / "assets" / "mjcf" / f"{robot_name}.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"Robot XML file not found at: {xml_path}")

    humanoid_mjcf = mjcf.from_path(str(xml_path))
    root.include_copy(humanoid_mjcf)

    root_str = to_string(root, pretty=True)
    return mujoco.MjModel.from_xml_string(root_str)


# ---- Core Retargeting Logic ----

def retarget_motion_with_mink(
    joint_positions: np.ndarray,
    joint_names: list,
    skeleton_tree: SkeletonTree,
    fps: int,
    render: bool = False
) -> SkeletonMotion:
    """
    Retargets motion using mink IK solver.
    """
    robot_type = "smpl_humanoid_lower_body"
    n_frames = joint_positions.shape[0]
    model_joint_names = skeleton_tree.node_names
    n_model_joints = len(model_joint_names)

    # Prepare global translations for all model joints from the input positions
    global_translations = np.zeros((n_frames, n_model_joints, 3))
    for i, name in enumerate(joint_names):
        if name in model_joint_names:
            model_idx = model_joint_names.index(name)
            global_translations[:, model_idx, :] = joint_positions[:, i, :]

    # Use identity quaternions for orientation targets as we only have position data
    pose_quat_global = np.zeros((n_frames, n_model_joints, 4))
    pose_quat_global[..., 0] = 1.0  # WXYZ format

    # --- Mink IK setup ---
    model = construct_model(robot_type, model_joint_names)
    configuration = mink.Configuration(model)
    tasks = []
    keypoint_map = _KEYPOINT_TO_JOINT_MAP[robot_type]

    for joint_name, retarget_info in keypoint_map.items():
        task = mink.FrameTask(
            frame_name=retarget_info["name"],
            frame_type="body",
            position_cost=10.0 * retarget_info["weight"],
            orientation_cost=0.0,  # No orientation data
            lm_damping=1.0,
        )
        tasks.append(task)

    posture_task = mink.PostureTask(model, cost=1.0)
    tasks.append(posture_task)

    data = configuration.data

    # --- Viewer setup (optional) ---
    viewer_context = mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=True) if render else nullcontext()

    all_root_translations = []
    all_root_rotations = []
    all_joint_angles = []
    all_joint_positions = []

    with viewer_context as viewer:
        if render and viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = [0, 0, 1]
            viewer.cam.distance = 3.0

        # Initialize pose
        data.qpos[0:3] = global_translations[0, 0]  # Initial root position
        data.qpos[3:7] = pose_quat_global[0, 0]  # Initial root orientation (identity)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        posture_task.set_target_from_configuration(configuration)

        solver = "quadprog"
        rate = RateLimiter(frequency=float(fps))
        pbar = tqdm(total=n_frames, desc="Retargeting frames")

        for t in range(n_frames):
            # Set IK targets for the current frame
            for i, (joint_name, retarget_info) in enumerate(keypoint_map.items()):
                model_idx = model_joint_names.index(joint_name)
                target_pos = global_translations[t, model_idx, :].copy()
                target_rot = pose_quat_global[t, model_idx].copy()

                rot_matrix = sRot.from_quat(np.roll(target_rot, -1)).as_matrix()  # WXYZ -> XYZW
                rot = mink.SO3.from_matrix(rot_matrix)
                tasks[i].set_target(mink.SE3.from_rotation_and_translation(rot, target_pos))

            # Solve IK
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver)
            configuration.integrate_inplace(vel, rate.dt)

            all_root_translations.append(configuration.data.qpos[:3].copy())
            all_root_rotations.append(configuration.data.qpos[3:7].copy())
            all_joint_angles.append(configuration.data.qpos[7:].copy())
            all_joint_positions.append(configuration.data.xpos[10:].copy())

            # print(f"[DEBUG] Frame {t}: Root Translation: {all_root_translations[-1]}, Global Rotation: {all_root_rotations[-1]}")
            # print(f"[DEBUG] Frame {t}: Joint Angles shape: {all_joint_angles[-1].shape}")
            # print(f"[DEBUG] Frame {t}: Joint Angles: {all_joint_angles[-1]}")
            # print(f"[DEBUG] Frame {t}: Joint Positions shape: {all_joint_positions[-1].shape}")
            # print(f"[DEBUG] Frame {t}: Joint Positions: {all_joint_positions[-1]}")

            pbar.update(1)
            if render and viewer:
                viewer.sync()
                rate.sleep()

        pbar.close()

    root_translations = np.array(all_root_translations)

    root_rotations_wxyz = all_root_rotations
    root_rotations_xyzw = np.roll(root_rotations_wxyz, shift=-1, axis=-1)

    rotations_xyz = all_joint_angles
    rotations_xyzw = euler_to_quaternion(rotations_xyz)
    # reshape the rotations_xyzw to (n_frames, n_joints, 4)
    rotations_xyzw = np.array(rotations_xyzw).reshape(n_frames, (len(model_joint_names)-1), 4)

    rotations_quat = np.zeros((n_frames, len(model_joint_names) * 4))


    for i in range(n_frames):
        # Concatenate root rotation and joint rotations into a single array
        rotations_quat[i] = np.concatenate((root_rotations_xyzw[i], rotations_xyzw[i].flatten()))
        # print(f"[DEBUG] Frame {i}: Rotations (Euler): {rotations_euler}")
    
    # reshape into (n_frames, n_joints, 4)
    rotations_quat = np.array(rotations_quat).reshape(n_frames, (len(model_joint_names)), 4)


    joint_positions = np.array(all_joint_positions)
    print(f"[DEBUG] Joint Positions: {joint_positions.shape}")
    print(f"[DEBUG] Root Translations: {root_translations.shape}")
    print(f"[DEBUG] Joint Quaternion Angles: {rotations_quat.shape}")



    # print(f"check")

    return root_translations, rotations_quat


def quaternion_to_euler(quaternion_data):
    """
    Converts a list of time frames containing quaternions to XYZ Euler angles.

    Args:
        quaternion_data (list of np.array): A list where each element is a NumPy
                                            array representing a time frame. Each
                                            array contains 32 float values
                                            (8 joints * 4 quaternion components [w, x, y, z]).

    Returns:
        list of np.array: A list where each element is a NumPy array of Euler angles
                          for a time frame. Each array will contain 24 float values
                          (8 joints * 3 XYZ angles in radians).
    """
    euler_frames = []

    # Iterate over each time frame in the input data
    for frame_quaternions in quaternion_data:
        # Reshape the flat array of 32 numbers into an 8x4 matrix (8 joints, 4 components each)
        joints = frame_quaternions.reshape(1, 4)
        
        frame_euler_angles = []

        # Iterate over each joint's quaternion
        for q in joints:
            w, x, y, z = q[0], q[1], q[2], q[3]

            # singularity check for Pitch (gimbal lock)
            # test for singularity at north pole
            sinp = 2 * (w * y - z * x)
            if np.abs(sinp) >= 1:
                pitch = np.copysign(np.pi / 2, sinp) # use pi/2
                
                # gimbal lock case
                roll = np.arctan2(x, w) * 2
                yaw = 0
            else:
                # Pitch (y-axis rotation)
                pitch = np.arcsin(sinp)
                
                # Roll (x-axis rotation)
                sinr_cosp = 2 * (w * x + y * z)
                cosr_cosp = 1 - 2 * (x**2 + y**2)
                roll = np.arctan2(sinr_cosp, cosr_cosp)
                
                # Yaw (z-axis rotation)
                siny_cosp = 2 * (w * z + x * y)
                cosy_cosp = 1 - 2 * (y**2 + z**2)
                yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            # Append the resulting XYZ Euler angles for the joint
            frame_euler_angles.extend([roll, pitch, yaw])
        
        # Convert the list of Euler angles for the frame into a single NumPy array
        # and add it to our list of processed frames
        euler_frames.append(np.array(frame_euler_angles))
        
    return euler_frames


def euler_to_quaternion(rotation_data):
    """
    Converts a list of time frames containing XYZ Euler angles to quaternions.

    Args:
        rotation_data (list of np.array): A list where each element is a NumPy array
                                           representing a time frame. Each array contains
                                           24 float values (8 joints * 3 XYZ angles in radians).

    Returns:
        list of np.array: A list where each element is a NumPy array of quaternions
                          for a time frame. Each array will contain 32 float values
                          (8 joints * 4 quaternion components [w, x, y, z]).
    """
    quaternion_frames = []

    # Iterate over each time frame in the input data
    for frame_euler_angles in rotation_data:
        # Reshape the flat array of 24 numbers into an 8x3 matrix (8 joints, 3 angles each)
        joints = frame_euler_angles.reshape(8, 3)
        
        frame_quaternions = []

        # Iterate over each joint's XYZ angles
        for joint_angles in joints:
            # Extract roll (x), pitch (y), and yaw (z) angles
            roll, pitch, yaw = joint_angles[0], joint_angles[1], joint_angles[2]

            # Calculate cosine and sine of half the angles
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)
            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)

            # Calculate quaternion components (w, x, y, z)
            # This is the standard formula for XYZ Euler (Tait-Bryan) to quaternion conversion
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy
            
            # Append the resulting quaternion for the joint
            frame_quaternions.extend([ x, y, z, w])
        
        # Convert the list of quaternions for the frame into a single NumPy array
        # and add it to our list of processed frames
        quaternion_frames.append(np.array(frame_quaternions))
        
    return quaternion_frames


def create_smpl_lower_body_skeleton_tree() -> SkeletonTree:
    """
    Create a skeleton tree for the SMPL lower body model using the provided XML file.
    """
    xml_path = Path(__file__).parent.parent.parent / "protomotions" / "data" / "assets" / "mjcf" / "smpl_humanoid_lower_body.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"SMPL lower body XML file not found at: {xml_path}")

    # Load directly; poselib's from_mjcf expects parent-local offsets.
    return SkeletonTree.from_mjcf(str(xml_path))

def create_skeleton_motion_from_retargeted_data(
    all_root_translations: np.ndarray,
    rotations_quat: np.ndarray,
    skeleton_tree: SkeletonTree,
    fps: int
) -> SkeletonMotion:
    """
    Converts retargeted motion data into a poselib.SkeletonMotion object.

    Args:
        all_root_translations (np.ndarray): Array of root translations for each frame.
        rotations_quat (np.ndarray): Array of joint rotations (as quaternions) for each frame.
        skeleton_tree (SkeletonTree): The skeleton tree for the character.
        fps (int): The frames per second of the motion.

    Returns:
        SkeletonMotion: The resulting SkeletonMotion object.
    """
    # Convert numpy arrays to torch tensors
    root_translations_tensor = torch.from_numpy(all_root_translations).float()
    rotations_tensor = torch.from_numpy(rotations_quat).float()

    # Create a SkeletonState object from the rotations and translations
    # We assume the rotations from your IK solver are global rotations, so is_local=False
    skeleton_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree=skeleton_tree,
        r=rotations_tensor,
        t=root_translations_tensor,
        is_local=True  # Assuming global rotations from IK
    )

    # Convert the SkeletonState to a SkeletonMotion object
    # This will also compute the joint velocities and angular velocities
    skeleton_motion = SkeletonMotion.from_skeleton_state(
        skeleton_state=skeleton_state,
        fps=fps
    )

    return skeleton_motion


def save_skeleton_motion(sk_motion: SkeletonMotion, output_path: str):
    """
    Save skeleton motion to file in the format expected by the data pipeline.
    """
    output_path_obj = Path(output_path)
    output_dir = output_path_obj.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    sk_motion.to_file(output_path)

    print(f"[SUCCESS] Saved skeleton motion to: {output_path}")
    print(f"   - FPS: {sk_motion.fps}")
    print(f"   - Frames: {sk_motion.global_translation.shape[0]}")
    print(f"   - Joints: {sk_motion.global_translation.shape[1]}")
    print(f"   - Is local: {sk_motion.is_local}")


def extract_sagittal_angles(sk_motion: SkeletonMotion) -> dict:
    """
    Extract sagittal plane angles from skeleton motion.
    """
    local_rotations = sk_motion.local_rotation.numpy()
    joint_names = sk_motion.skeleton_tree.node_names
    sagittal_angles = {}

    for joint_idx, joint_name in enumerate(joint_names):
        if joint_name == 'Pelvis':
            continue

        joint_quats_wxyz = local_rotations[:, joint_idx, :]
        joint_quats_xyzw = np.roll(joint_quats_wxyz, -1, axis=1)  # WXYZ -> XYZW

        # Normalize quaternions
        norms = np.linalg.norm(joint_quats_xyzw, axis=1, keepdims=True)
        # Create a mask for quaternions with significant norm
        non_zero_mask = (norms > 1e-6)

        # Get the norms for division, replacing zeros with 1 to avoid warnings/errors
        div_norms = np.where(non_zero_mask, norms, 1.0)

        # Normalize all quaternions
        joint_quats_xyzw = joint_quats_xyzw / div_norms

        # For quaternions that originally had a zero norm, set them to identity
        joint_quats_xyzw[~non_zero_mask.squeeze(), :] = np.array([0.0, 0.0, 0.0, 1.0])

        try:
            euler_angles = sRot.from_quat(joint_quats_xyzw).as_euler('xyz', degrees=True)
            sagittal_angles[joint_name] = euler_angles[:, 1]  # Y-axis for flexion/extension
        except ValueError:
            sagittal_angles[joint_name] = np.zeros(local_rotations.shape[0])

    return sagittal_angles


def plot_sagittal_angles(sk_motion: SkeletonMotion, save_path: Optional[str] = None):
    """
    Plot sagittal plane angles for each joint.
    """
    print("[PLOT] Extracting sagittal plane angles...")
    try:
        sagittal_angles = extract_sagittal_angles(sk_motion)
    except Exception as e:
        print(f"Error extracting sagittal angles: {e}")
        return

    n_frames = sk_motion.local_rotation.shape[0]
    time_axis = np.arange(n_frames) / sk_motion.fps

    joint_names = list(sagittal_angles.keys())
    n_joints = len(joint_names)
    cols = 3
    rows = (n_joints + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for i, joint_name in enumerate(joint_names):
        ax = axes_flat[i]
        angles = sagittal_angles[joint_name]
        ax.plot(time_axis, angles, linewidth=2, color='blue')
        ax.set_title(f'{joint_name} Sagittal Angle', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Angle (degrees)', fontsize=10)
        ax.grid(True, alpha=0.3)

        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        ax.text(0.02, 0.98, f'Mean: {mean_angle:.1f}°\nStd: {std_angle:.1f}°',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    for i in range(n_joints, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVE] Saved sagittal angle plot to: {save_path}")

    plt.show()


def main(
    input_file: str = typer.Argument(..., help="Input .npy file from treadmill2overground.py"),
    output_file: str = typer.Argument(..., help="Output file path for retargeted motion"),
    fps: int = typer.Option(200, "--fps", "-f", help="Frame rate of the motion data"),
    joint_names: Optional[str] = typer.Option(
        None, "--joints", "-j",
        help="Comma-separated joint names (default: Pelvis,L_Hip,L_Knee,L_Ankle,L_Toe,R_Hip,R_Knee,R_Ankle,R_Toe)"
    ),
    plot_angles: bool = typer.Option(False, "--plot", "-p", help="Plot sagittal plane angles after retargeting"),
    plot_motion: bool = typer.Option(False, "--plot-motion", "-pm", help="Plot the final skeleton motion in a 3D viewer"),
    plot_raw_motion: bool = typer.Option(False, "--plot-raw-motion", "-prm", help="Plot the raw skeleton motion in a 3D viewer before final processing"),
    plot_save_path: Optional[str] = typer.Option(None, "--plot-save", help="Path to save the angle plot (optional)"),
    render: bool = typer.Option(False, "--render", help="Render the retargeting process in a MuJoCo viewer.")
):
    """
    Retarget motion data from treadmill2overground.py output to SMPL lower body model format.
    """
    print("[RETARGET] Treadmill Motion Retargeting to SMPL Lower Body (using Mink)")
    print("=" * 60)

    if joint_names is None:
        joint_names = "Pelvis,L_Hip,L_Knee,L_Ankle,L_Toe,R_Hip,R_Knee,R_Ankle,R_Toe"

    joint_names_list = [name.strip() for name in joint_names.split(",")]

    try:
        print(f"Loading joint positions from {input_file}...")
        joint_positions = np.load(input_file)
        
        sk_tree = create_smpl_lower_body_skeleton_tree()

        print("[SUCCESS] Loaded motion data:")
        print(f"   - Shape: {joint_positions.shape}")
        print(f"   - Joint names: {joint_names_list}")

        if len(joint_names_list) != joint_positions.shape[1]:
            raise ValueError(f"Number of joint names ({len(joint_names_list)}) doesn't match number of joints in data ({joint_positions.shape[1]})")

        print("\n[RETARGET] Creating skeleton motion using mink for retargeting...")
        root_translations, rotations_quat = retarget_motion_with_mink(
            joint_positions, joint_names_list, sk_tree, fps=fps, render=render
        )

        sk_motion = create_skeleton_motion_from_retargeted_data(
            root_translations, rotations_quat, sk_tree, fps=fps
        )

        print("\n[SAVE] Saving retargeted motion...")
        save_skeleton_motion(sk_motion, output_file)

        if plot_angles:
            print("\n[PLOT] Plotting sagittal plane angles...")
            plot_sagittal_angles(sk_motion, plot_save_path)

        print("\n[SUCCESS] Retargeting completed successfully!")

    except Exception as e:
        print(f"[ERROR] Error during retargeting: {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
