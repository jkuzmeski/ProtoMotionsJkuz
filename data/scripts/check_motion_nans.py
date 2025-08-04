import numpy as np
import torch
from collections import OrderedDict

def check_motion_file(motion_file_path):
    """Check for NaN values in the motion file and print detailed information."""
    print(f"Checking motion file: {motion_file_path}")
    
    # Load the motion file
    motion_data = np.load(motion_file_path, allow_pickle=True).item()
    
    print("\n=== Motion File Structure ===")
    for key, value in motion_data.items():
        print(f"{key}: {type(value)}")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {type(subvalue)}")
                if hasattr(subvalue, 'shape'):
                    print(f"    Shape: {subvalue.shape}")
                    if hasattr(subvalue, 'dtype'):
                        print(f"    Dtype: {subvalue.dtype}")
    
    print("\n=== NaN Check ===")
    nan_found = False
    
    for key, value in motion_data.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if hasattr(subvalue, 'arr'):
                    arr = subvalue['arr']
                    if hasattr(arr, 'shape'):
                        nan_count = np.isnan(arr).sum()
                        inf_count = np.isinf(arr).sum()
                        if nan_count > 0 or inf_count > 0:
                            nan_found = True
                            print(f"❌ {key}.{subkey}: {nan_count} NaNs, {inf_count} Infs")
                            print(f"   Shape: {arr.shape}, Dtype: {arr.dtype}")
                            print(f"   Min: {np.nanmin(arr)}, Max: {np.nanmax(arr)}")
                            print(f"   Mean: {np.nanmean(arr)}")
                        else:
                            print(f"✅ {key}.{subkey}: No NaNs/Infs")
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            if isinstance(value, torch.Tensor):
                arr = value.numpy()
            else:
                arr = value
            nan_count = np.isnan(arr).sum()
            inf_count = np.isinf(arr).sum()
            if nan_count > 0 or inf_count > 0:
                nan_found = True
                print(f"❌ {key}: {nan_count} NaNs, {inf_count} Infs")
                print(f"   Shape: {arr.shape}, Dtype: {arr.dtype}")
                print(f"   Min: {np.nanmin(arr)}, Max: {np.nanmax(arr)}")
                print(f"   Mean: {np.nanmean(arr)}")
            else:
                print(f"✅ {key}: No NaNs/Infs")
    
    if not nan_found:
        print("✅ No NaN or Inf values found in motion file!")
    else:
        print("\n❌ NaN/Inf values found! Motion file needs to be fixed.")
    
    return nan_found

def fix_motion_file(motion_file_path, output_file_path=None):
    """Fix NaN values in the motion file by replacing them with zeros."""
    if output_file_path is None:
        output_file_path = motion_file_path.replace('.npy', '_fixed.npy')
    
    print(f"Fixing motion file: {motion_file_path}")
    print(f"Output file: {output_file_path}")
    
    # Load the motion file
    motion_data = np.load(motion_file_path, allow_pickle=True).item()
    
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
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            if isinstance(value, torch.Tensor):
                arr = value.numpy()
            else:
                arr = value
            arr_fixed = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            fixed_data[key] = arr_fixed
        else:
            fixed_data[key] = value
    
    # Save the fixed motion file
    np.save(output_file_path, fixed_data)
    print(f"✅ Fixed motion file saved to: {output_file_path}")
    
    # Verify the fix
    print("\n=== Verifying Fix ===")
    check_motion_file(output_file_path)
    
    return output_file_path

if __name__ == "__main__":
    motion_file = "data/motions/retargeted_motion.npy"
    
    # Check for NaNs
    has_nans = check_motion_file(motion_file)
    
    if has_nans:
        print("\n=== Fixing Motion File ===")
        fixed_file = fix_motion_file(motion_file)
        print(f"\nUse the fixed file: {fixed_file}")
    else:
        print("\nMotion file is clean!") 