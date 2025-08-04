#!/usr/bin/env python3
"""
Script to fix motion data for lower body configuration.
This script ensures the motion data is properly formatted and doesn't contain NaN values.
"""

import numpy as np

def fix_motion_data(input_file, output_file):
    """Fix motion data for lower body configuration."""
    print(f"Loading motion data from: {input_file}")
    
    # Load the motion data
    data = np.load(input_file, allow_pickle=True).item()
    
    print("Original data structure:")
    print(f"  Rotation shape: {data['rotation']['arr'].shape}")
    print(f"  Root translation shape: {data['root_translation']['arr'].shape}")
    print(f"  Global velocity shape: {data['global_velocity']['arr'].shape}")
    print(f"  Global angular velocity shape: {data['global_angular_velocity']['arr'].shape}")
    print(f"  Node names: {data['skeleton_tree']['node_names']}")
    
    # Check for NaN values
    rotation_nan = np.isnan(data['rotation']['arr']).sum()
    root_trans_nan = np.isnan(data['root_translation']['arr']).sum()
    global_vel_nan = np.isnan(data['global_velocity']['arr']).sum()
    global_ang_vel_nan = np.isnan(data['global_angular_velocity']['arr']).sum()
    
    print("\nNaN counts:")
    print(f"  Rotation: {rotation_nan}")
    print(f"  Root translation: {root_trans_nan}")
    print(f"  Global velocity: {global_vel_nan}")
    print(f"  Global angular velocity: {global_ang_vel_nan}")
    
    # Fix NaN values by replacing with zeros
    if rotation_nan > 0:
        print("  Fixing rotation NaNs...")
        data['rotation']['arr'] = np.nan_to_num(data['rotation']['arr'], nan=0.0, posinf=0.0, neginf=0.0)
    
    if root_trans_nan > 0:
        print("  Fixing root translation NaNs...")
        data['root_translation']['arr'] = np.nan_to_num(data['root_translation']['arr'], nan=0.0, posinf=0.0, neginf=0.0)
    
    if global_vel_nan > 0:
        print("  Fixing global velocity NaNs...")
        data['global_velocity']['arr'] = np.nan_to_num(data['global_velocity']['arr'], nan=0.0, posinf=0.0, neginf=0.0)
    
    if global_ang_vel_nan > 0:
        print("  Fixing global angular velocity NaNs...")
        data['global_angular_velocity']['arr'] = np.nan_to_num(data['global_angular_velocity']['arr'], nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize quaternions to ensure they are unit quaternions
    print("\nNormalizing quaternions...")
    rotation_arr = data['rotation']['arr']
    if len(rotation_arr.shape) >= 2 and rotation_arr.shape[-1] == 4:
        norms = np.linalg.norm(rotation_arr, axis=-1)
        # Avoid division by zero
        norms = np.clip(norms, 1e-6, None)
        rotation_arr_normalized = rotation_arr / norms[..., np.newaxis]
        data['rotation']['arr'] = rotation_arr_normalized
        print(f"  Quaternions normalized. Non-unit count before: {np.sum(np.abs(norms - 1.0) > 1e-6)}")
    
    # Bound extreme values to prevent numerical issues
    print("\nBounding extreme values...")
    
    # Root translation bounds
    root_trans = data['root_translation']['arr']
    root_trans = np.clip(root_trans, -100.0, 100.0)
    data['root_translation']['arr'] = root_trans
    
    # Global velocity bounds
    global_vel = data['global_velocity']['arr']
    global_vel = np.clip(global_vel, -50.0, 50.0)
    data['global_velocity']['arr'] = global_vel
    
    # Global angular velocity bounds
    global_ang_vel = data['global_angular_velocity']['arr']
    global_ang_vel = np.clip(global_ang_vel, -100.0, 100.0)
    data['global_angular_velocity']['arr'] = global_ang_vel
    
    # Verify the data structure is correct for lower body
    expected_nodes = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe']
    actual_nodes = data['skeleton_tree']['node_names']
    
    if actual_nodes != expected_nodes:
        print(f"\n⚠️  Warning: Node names don't match expected lower body configuration!")
        print(f"  Expected: {expected_nodes}")
        print(f"  Actual: {actual_nodes}")
    else:
        print(f"\n✅ Node names match expected lower body configuration")
    
    # Verify shapes are correct
    num_frames = data['rotation']['arr'].shape[0]
    num_joints = len(actual_nodes)
    
    expected_shapes = {
        'rotation': (num_frames, num_joints, 4),
        'root_translation': (num_frames, 3),
        'global_velocity': (num_frames, num_joints, 3),
        'global_angular_velocity': (num_frames, num_joints, 3)
    }
    
    print(f"\nVerifying shapes:")
    for key, expected_shape in expected_shapes.items():
        actual_shape = data[key]['arr'].shape
        if actual_shape == expected_shape:
            print(f"  ✅ {key}: {actual_shape}")
        else:
            print(f"  ❌ {key}: {actual_shape} (expected {expected_shape})")
    
    # Save the fixed motion data
    print(f"\nSaving fixed motion data to: {output_file}")
    np.save(output_file, data)
    
    # Verify the saved data
    print("\nVerifying saved data...")
    saved_data = np.load(output_file, allow_pickle=True).item()
    
    # Check for NaN values in saved data
    saved_rotation_nan = np.isnan(saved_data['rotation']['arr']).sum()
    saved_root_trans_nan = np.isnan(saved_data['root_translation']['arr']).sum()
    saved_global_vel_nan = np.isnan(saved_data['global_velocity']['arr']).sum()
    saved_global_ang_vel_nan = np.isnan(saved_data['global_angular_velocity']['arr']).sum()
    
    print(f"Saved data NaN counts:")
    print(f"  Rotation: {saved_rotation_nan}")
    print(f"  Root translation: {saved_root_trans_nan}")
    print(f"  Global velocity: {saved_global_vel_nan}")
    print(f"  Global angular velocity: {saved_global_ang_vel_nan}")
    
    if all(count == 0 for count in [saved_rotation_nan, saved_root_trans_nan, saved_global_vel_nan, saved_global_ang_vel_nan]):
        print("✅ Motion data successfully fixed and saved!")
    else:
        print("❌ Some NaN values still remain!")
    
    return output_file

def main():
    """Main function."""
    input_file = "data/motions/retargeted_motion.npy"
    output_file = "data/motions/retargeted_motion_fixed.npy"
    
    try:
        fixed_file = fix_motion_data(input_file, output_file)
        print(f"\n🎉 Fixed motion data saved to: {fixed_file}")
        print("You can now use this file for evaluation.")
    except Exception as e:
        print(f"❌ Error fixing motion data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 