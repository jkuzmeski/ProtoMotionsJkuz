#!/usr/bin/env python3
"""
Script to compare the structure of motion files.
Compares retargeted_motion.npy with smpl_humanoid_walk.npy to ensure they have the same structure.
Also checks for NaN, inf, and -inf values in the motion data.
"""

import numpy as np
import os
import sys
from collections import OrderedDict


def check_for_nan_values(data, path="", max_reports=10):
    """Recursively check for NaN, inf, and -inf values in data structures."""
    nan_report = {
        'has_nan': False,
        'has_inf': False,
        'has_neg_inf': False,
        'nan_count': 0,
        'inf_count': 0,
        'neg_inf_count': 0,
        'nan_locations': [],
        'inf_locations': [],
        'neg_inf_locations': [],
        'total_elements': 0
    }
    
    def check_array(arr, current_path):
        if not isinstance(arr, np.ndarray):
            return
            
        nan_report['total_elements'] += arr.size
        
        # Check for NaN values
        nan_mask = np.isnan(arr)
        nan_count = np.sum(nan_mask)
        if nan_count > 0:
            nan_report['has_nan'] = True
            nan_report['nan_count'] += nan_count
            nan_indices = np.where(nan_mask)
            for i in range(min(len(nan_indices[0]), max_reports)):
                location = tuple(idx[i] for idx in nan_indices)
                nan_report['nan_locations'].append(f"{current_path}[{location}]")
        
        # Check for positive infinity
        inf_mask = np.isinf(arr) & (arr > 0)
        inf_count = np.sum(inf_mask)
        if inf_count > 0:
            nan_report['has_inf'] = True
            nan_report['inf_count'] += inf_count
            inf_indices = np.where(inf_mask)
            for i in range(min(len(inf_indices[0]), max_reports)):
                location = tuple(idx[i] for idx in inf_indices)
                nan_report['inf_locations'].append(f"{current_path}[{location}]")
        
        # Check for negative infinity
        neg_inf_mask = np.isinf(arr) & (arr < 0)
        neg_inf_count = np.sum(neg_inf_mask)
        if neg_inf_count > 0:
            nan_report['has_neg_inf'] = True
            nan_report['neg_inf_count'] += neg_inf_count
            neg_inf_indices = np.where(neg_inf_mask)
            for i in range(min(len(neg_inf_indices[0]), max_reports)):
                location = tuple(idx[i] for idx in neg_inf_indices)
                nan_report['neg_inf_locations'].append(f"{current_path}[{location}]")
    
    def recursive_check(obj, current_path):
        if isinstance(obj, np.ndarray):
            check_array(obj, current_path)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{current_path}['{key}']" if current_path else f"['{key}']"
                recursive_check(value, new_path)
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                new_path = f"{current_path}[{i}]" if current_path else f"[{i}]"
                recursive_check(item, new_path)
        elif hasattr(obj, '__dict__'):
            for attr_name, attr_value in obj.__dict__.items():
                new_path = f"{current_path}.{attr_name}" if current_path else f".{attr_name}"
                recursive_check(attr_value, new_path)
    
    recursive_check(data, path)
    return nan_report


def print_nan_report(nan_report, file_name):
    """Print a detailed report of NaN and infinity values found."""
    print(f"\n=== NaN/Inf Report for {file_name} ===")
    
    if nan_report['has_nan']:
        print(f"❌ Found {nan_report['nan_count']} NaN values")
        print("   First few locations:")
        for loc in nan_report['nan_locations'][:5]:
            print(f"     {loc}")
        if len(nan_report['nan_locations']) > 5:
            print(f"     ... and {len(nan_report['nan_locations']) - 5} more")
    
    if nan_report['has_inf']:
        print(f"❌ Found {nan_report['inf_count']} positive infinity values")
        print("   First few locations:")
        for loc in nan_report['inf_locations'][:5]:
            print(f"     {loc}")
        if len(nan_report['inf_locations']) > 5:
            print(f"     ... and {len(nan_report['inf_locations']) - 5} more")
    
    if nan_report['has_neg_inf']:
        print(f"❌ Found {nan_report['neg_inf_count']} negative infinity values")
        print("   First few locations:")
        for loc in nan_report['neg_inf_locations'][:5]:
            print(f"     {loc}")
        if len(nan_report['neg_inf_locations']) > 5:
            print(f"     ... and {len(nan_report['neg_inf_locations']) - 5} more")
    
    if not (nan_report['has_nan'] or nan_report['has_inf'] or nan_report['has_neg_inf']):
        print(f"✅ No NaN or infinity values found")
    
    print(f"   Total elements checked: {nan_report['total_elements']:,}")


def analyze_dictionary_structure(dictionary, indent="  "):
    """Analyze and print the structure of a dictionary."""
    print(f"{indent}Dictionary structure:")
    for key, value in dictionary.items():
        if isinstance(value, dict):
            print(f"{indent}  {key}: dict with {len(value)} keys")
            for subkey, subvalue in value.items():
                if hasattr(subvalue, 'shape'):
                    print(f"{indent}    {subkey}: array shape {subvalue.shape}, dtype {subvalue.dtype}")
                else:
                    print(f"{indent}    {subkey}: {type(subvalue)}")
        elif hasattr(value, 'shape'):
            print(f"{indent}  {key}: array shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"{indent}  {key}: {type(value)}")


def print_skeleton_info(motion_data, file_name):
    """Print detailed skeleton information."""
    if 'skeleton_tree' in motion_data:
        skeleton = motion_data['skeleton_tree']
        print(f"\n=== Skeleton Information for {file_name} ===")
        
        if 'node_names' in skeleton:
            print(f"\nNode names ({len(skeleton['node_names'])} nodes):")
            for i, name in enumerate(skeleton['node_names']):
                print(f"  {i}: {name}")
        
        if 'parent_indices' in skeleton:
            print("\nParent indices:")
            for node_name, parent_idx in skeleton['parent_indices'].items():
                print(f"  {node_name} -> {parent_idx}")
        
        if 'local_translation' in skeleton:
            print("\nLocal translations:")
            for node_name, translation in skeleton['local_translation'].items():
                print(f"  {node_name}: {translation}")


def debug_motion_processing(motion_data, file_name):
    """Debug the motion processing to find potential NaN sources."""
    print(f"\n=== Motion Processing Debug for {file_name} ===")
    
    # Check rotation data
    if 'rotation' in motion_data and 'arr' in motion_data['rotation']:
        rotation_arr = motion_data['rotation']['arr']
        print("\nRotation data:")
        print(f"  Shape: {rotation_arr.shape}")
        
        # Check quaternion norms
        if len(rotation_arr.shape) >= 2 and rotation_arr.shape[-1] == 4:
            norms = np.linalg.norm(rotation_arr, axis=-1)
            print(f"  Quaternion norms - Min: {np.min(norms):.6f}, Max: {np.max(norms):.6f}")
            print(f"  Quaternion norms - Mean: {np.mean(norms):.6f}")
            
            # Check for quaternions that are not unit quaternions
            non_unit_count = np.sum(np.abs(norms - 1.0) > 1e-6)
            print(f"  Non-unit quaternions: {non_unit_count}")
            
            if non_unit_count > 0:
                print("  ⚠️  Found non-unit quaternions! This could cause NaN issues.")
                # Normalize quaternions
                rotation_arr_normalized = rotation_arr / norms[..., np.newaxis]
                print("  ✅ Quaternions normalized")
                return rotation_arr_normalized
            else:
                print("  ✅ All quaternions are unit quaternions")
                return rotation_arr
    
    # Check angular velocity
    if 'global_angular_velocity' in motion_data and 'arr' in motion_data['global_angular_velocity']:
        angular_velocity_arr = motion_data['global_angular_velocity']['arr']
        print("\nAngular velocity data:")
        print(f"  Shape: {angular_velocity_arr.shape}")
        
        # Check for extreme values in angular velocity
        ang_vel_magnitude = np.linalg.norm(angular_velocity_arr, axis=-1)
        print(f"  Angular velocity magnitude - Min: {np.min(ang_vel_magnitude):.6f}, Max: {np.max(ang_vel_magnitude):.6f}")
        print(f"  Angular velocity magnitude - Mean: {np.mean(ang_vel_magnitude):.6f}")
        
        # Check for very large angular velocities
        large_ang_vel_count = np.sum(ang_vel_magnitude > 100.0)
        print(f"  Large angular velocities (>100): {large_ang_vel_count}")
    
    # Check global velocity
    if 'global_velocity' in motion_data and 'arr' in motion_data['global_velocity']:
        global_velocity_arr = motion_data['global_velocity']['arr']
        print("\nGlobal velocity data:")
        print(f"  Shape: {global_velocity_arr.shape}")
        
        # Check for extreme values in global velocity
        global_vel_magnitude = np.linalg.norm(global_velocity_arr, axis=-1)
        print(f"  Global velocity magnitude - Min: {np.min(global_vel_magnitude):.6f}, Max: {np.max(global_vel_magnitude):.6f}")
        print(f"  Global velocity magnitude - Mean: {np.mean(global_vel_magnitude):.6f}")
        
        # Check for very large velocities
        large_vel_count = np.sum(global_vel_magnitude > 10.0)
        print(f"  Large velocities (>10): {large_vel_count}")
    
    # Check root translation
    if 'root_translation' in motion_data and 'arr' in motion_data['root_translation']:
        root_translation_arr = motion_data['root_translation']['arr']
        print("\nRoot translation data:")
        print(f"  Shape: {root_translation_arr.shape}")
        
        # Check for extreme values in root translation
        print(f"  Root translation - Min: {np.min(root_translation_arr):.6f}, Max: {np.max(root_translation_arr):.6f}")
        print(f"  Root translation - Mean: {np.mean(root_translation_arr):.6f}")
        
        # Check for very large translations
        large_trans_count = np.sum(np.abs(root_translation_arr) > 100.0)
        print(f"  Large translations (>100): {large_trans_count}")
    
    return None


def create_clean_motion_file(motion_data, output_file_path=None):
    """Create a clean motion file with normalized quaternions and bounded velocities."""
    if output_file_path is None:
        output_file_path = "cleaned_motion.npy"
    
    print(f"\n=== Creating Clean Motion File ===")
    print(f"Output file: {output_file_path}")
    
    # Create clean motion data
    clean_data = OrderedDict()
    
    # Copy all non-array data
    for key, value in motion_data.items():
        if key not in ['rotation', 'root_translation', 'global_velocity', 'global_angular_velocity']:
            clean_data[key] = value
    
    # Clean rotation (normalize quaternions)
    if 'rotation' in motion_data and 'arr' in motion_data['rotation']:
        rotation_arr = motion_data['rotation']['arr']
        if len(rotation_arr.shape) >= 2 and rotation_arr.shape[-1] == 4:
            norms = np.linalg.norm(rotation_arr, axis=-1)
            # Avoid division by zero
            norms = np.clip(norms, 1e-6, None)
            rotation_arr_normalized = rotation_arr / norms[..., np.newaxis]
            print("✅ Quaternions normalized")
        else:
            rotation_arr_normalized = rotation_arr
        
        clean_data['rotation'] = {
            'arr': rotation_arr_normalized,
            'context': motion_data['rotation'].get('context', {})
        }
    
    # Clean root translation (bound extreme values)
    if 'root_translation' in motion_data and 'arr' in motion_data['root_translation']:
        root_translation_arr = motion_data['root_translation']['arr']
        root_translation_arr = np.clip(root_translation_arr, -100.0, 100.0)
        print("✅ Root translation bounded")
        clean_data['root_translation'] = {
            'arr': root_translation_arr,
            'context': motion_data['root_translation'].get('context', {})
        }
    
    # Clean global velocity (bound extreme values)
    if 'global_velocity' in motion_data and 'arr' in motion_data['global_velocity']:
        global_velocity_arr = motion_data['global_velocity']['arr']
        global_velocity_arr = np.clip(global_velocity_arr, -50.0, 50.0)
        print("✅ Global velocity bounded")
        clean_data['global_velocity'] = {
            'arr': global_velocity_arr,
            'context': motion_data['global_velocity'].get('context', {})
        }
    
    # Clean angular velocity (bound extreme values)
    if 'global_angular_velocity' in motion_data and 'arr' in motion_data['global_angular_velocity']:
        angular_velocity_arr = motion_data['global_angular_velocity']['arr']
        angular_velocity_arr = np.clip(angular_velocity_arr, -100.0, 100.0)
        print("✅ Angular velocity bounded")
        clean_data['global_angular_velocity'] = {
            'arr': angular_velocity_arr,
            'context': motion_data['global_angular_velocity'].get('context', {})
        }
    
    # Save the clean motion file
    np.save(output_file_path, clean_data)
    print(f"✅ Clean motion file saved to: {output_file_path}")
    
    return output_file_path


def fix_motion_file(motion_data, output_file_path=None):
    """Fix NaN values in the motion data by replacing them with zeros."""
    if output_file_path is None:
        output_file_path = "fixed_motion.npy"
    
    print(f"\n=== Fixing Motion File ===")
    print(f"Output file: {output_file_path}")
    
    # Fix NaN values
    fixed_data = OrderedDict()
    for key, value in motion_data.items():
        if isinstance(value, dict):
            fixed_dict = {}
            for subkey, subvalue in value.items():
                if hasattr(subvalue, 'arr'):
                    # Fix the array
                    arr = subvalue['arr']
                    if hasattr(arr, 'shape'):
                        arr_fixed = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                        fixed_dict[subkey] = {'arr': arr_fixed, 'context': subvalue.get('context', {})}
                    else:
                        fixed_dict[subkey] = subvalue
                else:
                    fixed_dict[subkey] = subvalue
            fixed_data[key] = fixed_dict
        elif isinstance(value, (np.ndarray, list)):
            if isinstance(value, list):
                arr = np.array(value)
            else:
                arr = value
            arr_fixed = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            fixed_data[key] = arr_fixed
        else:
            fixed_data[key] = value
    
    # Save the fixed motion file
    np.save(output_file_path, fixed_data)
    print(f"✅ Fixed motion file saved to: {output_file_path}")
    
    return output_file_path


def analyze_motion_file(file_path):
    """Analyze a single motion file in detail."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {file_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        motion_data = np.load(file_path, allow_pickle=True)
        print(f"✅ File loaded successfully")
        print(f"   Data type: {type(motion_data)}")
        print(f"   Shape: {motion_data.shape}")
        print(f"   Dtype: {motion_data.dtype}")
        
        # Check for NaN values
        nan_report = check_for_nan_values(motion_data, f"file['{os.path.basename(file_path)}']")
        if nan_report['has_nan'] or nan_report['has_inf'] or nan_report['has_neg_inf']:
            print(f"  ⚠️  Found problematic values!")
            print_nan_report(nan_report, os.path.basename(file_path))
        else:
            print(f"  ✅ No NaN/inf values found")
        
        # Handle 0-dimensional object arrays
        if motion_data.shape == () and motion_data.dtype == np.dtype('O'):
            print("\n0-dimensional object array detected - extracting single object:")
            single_object = motion_data.item()
            print(f"  Object type: {type(single_object)}")
            
            if isinstance(single_object, dict):
                print(f"  Dictionary with {len(single_object)} keys")
                
                # Check for NaN values in the dictionary
                dict_nan_report = check_for_nan_values(single_object, f"dict['{os.path.basename(file_path)}']")
                if dict_nan_report['has_nan'] or dict_nan_report['has_inf'] or dict_nan_report['has_neg_inf']:
                    print(f"    ⚠️  Found problematic values in dictionary!")
                    print_nan_report(dict_nan_report, f"dict['{os.path.basename(file_path)}']")
                
                analyze_dictionary_structure(single_object, indent="    ")
                
                # Print skeleton information
                print_skeleton_info(single_object, os.path.basename(file_path))
                
                # Debug motion processing
                debug_motion_processing(single_object, os.path.basename(file_path))
                
                return single_object
            
            else:
                print(f"  Unknown object type: {type(single_object)}")
                return single_object
        
        # If it's a structured array or object array, analyze further
        elif motion_data.dtype.names:
            print("\nStructured array fields:")
            for field_name in motion_data.dtype.names:
                field_data = motion_data[field_name]
                print(f"  - {field_name}: shape={field_data.shape}, dtype={field_data.dtype}")
                
                # Check for NaN values in each field
                field_nan_report = check_for_nan_values(field_data, f"field['{field_name}']")
                if field_nan_report['has_nan'] or field_nan_report['has_inf'] or field_nan_report['has_neg_inf']:
                    print(f"    ⚠️  Found problematic values in {field_name}!")
                    print_nan_report(field_nan_report, f"{field_name}")
                
                if len(field_data.shape) == 1:
                    print(f"    First few values: {field_data[:3]}")
                elif len(field_data.shape) == 2:
                    print(f"    Shape: {field_data.shape[0]} frames x {field_data.shape[1]} features")
                    print(f"    First frame: {field_data[0][:5]}...")
        
        # If it's an object array, check the first element
        elif motion_data.dtype == np.dtype('O'):
            print("\nObject array - analyzing first element:")
            if len(motion_data) > 0:
                first_element = motion_data[0]
                print(f"  First element type: {type(first_element)}")
                if isinstance(first_element, dict):
                    print(f"  Dictionary with {len(first_element)} keys")
                    analyze_dictionary_structure(first_element, indent="    ")
        
        # If it's a regular array, show basic info
        else:
            print(f"\nRegular array:")
            print(f"  Min: {np.min(motion_data)}")
            print(f"  Max: {np.max(motion_data)}")
            print(f"  Mean: {np.mean(motion_data)}")
            print(f"  Std: {np.std(motion_data)}")
            
            # Check for NaN values
            nan_count = np.isnan(motion_data).sum()
            inf_count = np.isinf(motion_data).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"  ⚠️  Found {nan_count} NaNs and {inf_count} Infs")
        
        return motion_data
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


def compare_motion_structures(file1_path, file2_path):
    """Compare the structures of two motion files."""
    print(f"\n{'='*80}")
    print(f"COMPARING MOTION STRUCTURES")
    print(f"{'='*80}")
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")
    
    # Analyze both files
    data1 = analyze_motion_file(file1_path)
    data2 = analyze_motion_file(file2_path)
    
    if data1 is None or data2 is None:
        print("❌ Cannot compare files - one or both failed to load")
        return
    
    print(f"\n{'='*60}")
    print(f"STRUCTURE COMPARISON")
    print(f"{'='*60}")
    
    # Compare basic properties
    print(f"Data types:")
    print(f"  File 1: {type(data1)}")
    print(f"  File 2: {type(data2)}")
    
    if hasattr(data1, 'shape') and hasattr(data2, 'shape'):
        print(f"Shapes:")
        print(f"  File 1: {data1.shape}")
        print(f"  File 2: {data2.shape}")
    
    if hasattr(data1, 'dtype') and hasattr(data2, 'dtype'):
        print(f"Dtypes:")
        print(f"  File 1: {data1.dtype}")
        print(f"  File 2: {data2.dtype}")
    
    # If both are dictionaries, compare keys
    if isinstance(data1, dict) and isinstance(data2, dict):
        keys1 = set(data1.keys())
        keys2 = set(data2.keys())
        
        print(f"\nDictionary keys:")
        print(f"  File 1 keys ({len(keys1)}): {sorted(keys1)}")
        print(f"  File 2 keys ({len(keys2)}): {sorted(keys2)}")
        
        common_keys = keys1 & keys2
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        
        print(f"\nKey comparison:")
        print(f"  Common keys ({len(common_keys)}): {sorted(common_keys)}")
        if only_in_1:
            print(f"  Only in File 1 ({len(only_in_1)}): {sorted(only_in_1)}")
        if only_in_2:
            print(f"  Only in File 2 ({len(only_in_2)}): {sorted(only_in_2)}")
        
        # Compare common keys in detail
        if common_keys:
            print(f"\nDetailed comparison of common keys:")
            for key in sorted(common_keys):
                val1 = data1[key]
                val2 = data2[key]
                
                print(f"\n  Key: {key}")
                print(f"    File 1 type: {type(val1)}")
                print(f"    File 2 type: {type(val2)}")
                
                if isinstance(val1, dict) and isinstance(val2, dict):
                    subkeys1 = set(val1.keys())
                    subkeys2 = set(val2.keys())
                    print(f"    File 1 subkeys: {sorted(subkeys1)}")
                    print(f"    File 2 subkeys: {sorted(subkeys2)}")
                    
                    if subkeys1 != subkeys2:
                        print(f"    ⚠️  Subkey mismatch!")
                        common_subkeys = subkeys1 & subkeys2
                        if common_subkeys:
                            print(f"    Common subkeys: {sorted(common_subkeys)}")
                
                elif hasattr(val1, 'shape') and hasattr(val2, 'shape'):
                    print(f"    File 1 shape: {val1.shape}")
                    print(f"    File 2 shape: {val2.shape}")
                    
                    if val1.shape != val2.shape:
                        print(f"    ⚠️  Shape mismatch!")
                    else:
                        # Compare values if shapes match
                        if hasattr(val1, 'dtype') and hasattr(val2, 'dtype'):
                            if val1.dtype == val2.dtype:
                                diff = np.abs(val1 - val2)
                                max_diff = np.max(diff)
                                mean_diff = np.mean(diff)
                                print(f"    Max difference: {max_diff:.6f}")
                                print(f"    Mean difference: {mean_diff:.6f}")
                            else:
                                print(f"    ⚠️  Dtype mismatch: {val1.dtype} vs {val2.dtype}")


def main():
    """Main function to analyze and compare motion files."""

    file1_path = 'D:\\Isaac\\IsaacLab2.1.2\\ProtoMotions\\data\\motions\\retargeted_motion.npy'
    file2_path = 'D:\\Isaac\\IsaacLab2.1.2\\ProtoMotions\\data\\motions\\smpl_humanoid_walk.npy'

    # Analyze first file
    data1 = analyze_motion_file(file1_path)
    
    if data1 is None:
        print("❌ Failed to analyze first file")
        return
    
    # Compare with second file if provided
    if file2_path:
        compare_motion_structures(file1_path, file2_path)


if __name__ == "__main__":
    main()
