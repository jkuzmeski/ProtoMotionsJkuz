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
from poselib.core.rotation3d import quat_inverse, quat_mul_norm, quat_from_angle_axis


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
    Retargets motion using mink IK solver and returns a SkeletonMotion object.
    """
    # ... (all your setup code remains the same until the main loop) ...
    robot_type = "smpl_humanoid_lower_body"
    n_frames = joint_positions.shape[0]
    model_joint_names = skeleton_tree.node_names
    n_model_joints = len(model_joint_names)

    global_translations = np.zeros((n_frames, n_model_joints, 3))
    for i, name in enumerate(joint_names):
        if name in model_joint_names:
            model_idx = model_joint_names.index(name)
            global_translations[:, model_idx, :] = joint_positions[:, i, :]

    pose_quat_global = np.zeros((n_frames, n_model_joints, 4))
    pose_quat_global[..., 0] = 1.0

    model = construct_model(robot_type, model_joint_names)
    configuration = mink.Configuration(model)
    tasks = []
    keypoint_map = _KEYPOINT_TO_JOINT_MAP[robot_type]

    for joint_name, retarget_info in keypoint_map.items():
        task = mink.FrameTask(
            frame_name=retarget_info["name"],
            frame_type="body",
            position_cost=10.0 * retarget_info["weight"],
            orientation_cost=0.1,
            lm_damping=1.0,
        )
        tasks.append(task)

    posture_task = mink.PostureTask(model, cost=1e-6)
    tasks.append(posture_task)
    data = configuration.data
    viewer_context = mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=True) if render else nullcontext()

    all_root_translations = []
    all_global_rotations = []

    with viewer_context as viewer:
        if render and viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = [0, 0, 1]
            viewer.cam.distance = 3.0

        data.qpos[0:3] = global_translations[0, 0]
        data.qpos[3:7] = pose_quat_global[0, 0]
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        # We still set the initial target here
        posture_task.set_target_from_configuration(configuration)

        solver = "quadprog"
        rate = RateLimiter(frequency=float(fps))
        pbar = tqdm(total=n_frames, desc="Retargeting frames")
        posture_task.set_target_from_configuration(configuration)
        
        for t in range(n_frames):
            for i, (joint_name, retarget_info) in enumerate(keypoint_map.items()):
                model_idx = model_joint_names.index(joint_name)
                target_pos = global_translations[t, model_idx, :].copy()
                target_rot = pose_quat_global[t, model_idx].copy()
                rot_matrix = sRot.from_quat(np.roll(target_rot, -1)).as_matrix()
                rot = mink.SO3.from_matrix(rot_matrix)
                
                tasks[i].set_target(mink.SE3.from_rotation_and_translation(rot, target_pos))

            vel = mink.solve_ik(configuration, tasks, rate.dt, solver)
            configuration.integrate_inplace(vel, rate.dt)

            all_root_translations.append(configuration.data.qpos[:3].copy())
            all_global_rotations.append(configuration.data.xquat.copy())
            print(f"[DEBUG] Frame {t}: Root Translation: {all_root_translations[-1]}, Global Rotation: {all_global_rotations[-1]}")

            pbar.update(1)
            if render and viewer:
                viewer.sync()
                rate.sleep()
        pbar.close()

    # ... (the rest of your function for creating the SkeletonMotion object is correct) ...
    root_translations_tensor = torch.from_numpy(np.stack(all_root_translations)).float()
    global_rotations_tensor_wxyz = torch.from_numpy(np.stack(all_global_rotations)).float()
    
    global_rotations_tensor = torch.roll(global_rotations_tensor_wxyz, shifts=-1, dims=-1)

    parent_indices = skeleton_tree.parent_indices
    local_rotations_list = []

    for i in range(n_model_joints):
        parent_idx = parent_indices[i]
        if parent_idx == -1:
            local_rotations_list.append(global_rotations_tensor[:, i + 1, :])
        else:
            parent_global_rot = global_rotations_tensor[:, parent_idx + 1, :]
            child_global_rot = global_rotations_tensor[:, i + 1, :]
            local_rot = quat_mul_norm(quat_inverse(parent_global_rot), child_global_rot)
            local_rotations_list.append(local_rot)
            
    local_rotations_tensor = torch.stack(local_rotations_list, dim=1)
    
    print("[INFO] Applying Z-up coordinate system correction to root rotation.")
    correction_quat = quat_from_angle_axis(
        angle=torch.tensor(-180.0),
        axis=torch.tensor([1.0, 0.0, 0.0]),
        degree=True
    )

    root_rotations = local_rotations_tensor[:, 0, :]
    corrected_root_rotations = quat_mul_norm(
        correction_quat.expand_as(root_rotations),
        root_rotations
    )
    local_rotations_tensor[:, 0, :] = corrected_root_rotations

    skel_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree=skeleton_tree,
        r=local_rotations_tensor,
        t=root_translations_tensor,
        is_local=True
    )

    skel_motion = SkeletonMotion.from_skeleton_state(skel_state, fps=fps)

    return skel_motion


# # CORRECT mapping from qpos indices to our skeleton joints
    # mujoco_to_skeleton_mapping = {
    #     # MuJoCo qpos[7:] order -> Skeleton joint index
    #     'L_Hip': (0, 1),    # qpos[7:10] -> skeleton joint 1 (L_Hip)
    #     'L_Knee': (3, 2),    # qpos[10:13] -> skeleton joint 2 (L_Knee)
    #     'L_Ankle': (6, 3),   # qpos[13:16] -> skeleton joint 3 (L_Ankle)
    #     'L_Toe': (9, 4),     # qpos[16:19] -> skeleton joint 4 (L_Toe)
    #     'R_Hip': (12, 5),    # qpos[19:22] -> skeleton joint 5 (R_Hip)
    #     'R_Knee': (15, 6),   # qpos[22:25] -> skeleton joint 6 (R_Knee)
    #     'R_Ankle': (18, 7),  # qpos[25:28] -> skeleton joint 7 (R_Ankle)
    #     'R_Toe': (21, 8),   # qpos[28:31] -> skeleton joint 8 (R_Toe)
    # }
    # Save the MuJoCo intermediate data for debugging
    
    # np.save('debug_mujoco_trans.npy', retargeted_trans)
    # np.save('debug_mujoco_poses.npy', retargeted_poses)


def create_smpl_lower_body_skeleton_tree() -> SkeletonTree:
    """
    Create a skeleton tree for the SMPL lower body model using the provided XML file.
    """
    xml_path = Path(__file__).parent.parent.parent / "protomotions" / "data" / "assets" / "mjcf" / "smpl_humanoid_lower_body.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"SMPL lower body XML file not found at: {xml_path}")

    # Load directly; poselib's from_mjcf expects parent-local offsets.
    return SkeletonTree.from_mjcf(str(xml_path))


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

        print("[SUCCESS] Loaded motion data:")
        print(f"   - Shape: {joint_positions.shape}")
        print(f"   - Joint names: {joint_names_list}")

        sk_tree = create_smpl_lower_body_skeleton_tree()

        if len(joint_names_list) != joint_positions.shape[1]:
            raise ValueError(f"Number of joint names ({len(joint_names_list)}) doesn't match number of joints in data ({joint_positions.shape[1]})")

        print("\n[RETARGET] Creating skeleton motion using mink for retargeting...")
        sk_motion = retarget_motion_with_mink(
            joint_positions, joint_names_list, sk_tree, fps=fps, render=render
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
