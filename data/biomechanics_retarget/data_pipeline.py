import os
import subprocess
import argparse
import sys
import glob

def run_command(command):
    """Executes a command in the shell and checks for errors."""
    try:
        command_str = ' '.join(f'"{c}"' if ' ' in c else c for c in command)
        print(f"Running command: {command_str}")
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        if result.stdout:
            print("Command output:\n" + result.stdout)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR ---")
        print(f"Error executing command: {' '.join(e.cmd)}")
        print(f"This command failed with return code: {e.returncode}")
        print("\n--- STDOUT from the failed script ---")
        print(e.stdout)
        print("\n--- STDERR from the failed script ---")
        print(e.stderr)
        print("\n--- End of error report ---")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Master batch pipeline for treadmill data processing.")
    parser.add_argument("--input_dir", required=True, help="Path to the directory containing raw treadmill data files.")
    parser.add_argument("--output_dir", required=True, help="Root directory to save all output files.")
    parser.add_argument("--file_pattern", default="*.txt", help="File pattern to match input files (e.g., '*.csv', 'subject*.txt').")
    args = parser.parse_args()
    
    # --- SOLUTION: Get the directory where this pipeline script is located ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory detected: {script_dir}")

    # --- SOLUTION: Define the full paths to your other scripts ---
    treadmill2overground_script = os.path.join(script_dir, "treadmill2overground.py")
    retarget_motion_script = os.path.join(script_dir, "retarget_treadmill_motion.py") # Corrected name to match uploaded file
    package_motion_script = os.path.join(script_dir, "package_motion_lib.py") # Corrected name to match uploaded file

    # --- 1. Setup directories ---
    overground_dir = os.path.join(args.output_dir, "overground_output")
    retargeted_dir = os.path.join(args.output_dir, "retargeted_output")
    packaged_dir = os.path.join(args.output_dir, "packaged_output")

    os.makedirs(overground_dir, exist_ok=True)
    os.makedirs(retargeted_dir, exist_ok=True)
    os.makedirs(packaged_dir, exist_ok=True)

    # --- 2. Find and process each input file ---
    search_path = os.path.join(args.input_dir, args.file_pattern)
    input_files = sorted(glob.glob(search_path))

    if not input_files:
        print(f"No files found matching '{search_path}'. Please check your --input_dir and --file_pattern.")
        sys.exit(0)

    print(f"Found {len(input_files)} files to process.")

    for i, input_file_path in enumerate(input_files):
        print(f"\n--- Processing file {i+1}/{len(input_files)}: {os.path.basename(input_file_path)} ---")
        base_name = os.path.splitext(os.path.basename(input_file_path))[0]
        overground_output_file = os.path.join(overground_dir, f"{base_name}_overground.npy")
        retargeted_output_file = os.path.join(retargeted_dir, f"{base_name}_retargeted.npy")

        # Step A: Run treadmill2overground.py
        print(f"\n[Step 1/2 for {base_name}] Converting to overground...")
        treadmill2overground_cmd = [
            "python", treadmill2overground_script,  # Use the full path
            input_file_path,
            overground_output_file
        ]
        run_command(treadmill2overground_cmd)

        # Step B: Run retarget_treadmill_motion.py
        print(f"\n[Step 2/2 for {base_name}] Retargeting motion...")
        retargeted_cmd = [
            "python", retarget_motion_script, # Use the full path
            "--input_file", overground_output_file,
            "--output_file", retargeted_output_file
        ]
        run_command(retargeted_cmd)

    # --- 3. Final packaging step ---
    print("\n--- All files processed. Starting final packaging step. ---")
    package_cmd = [
        "python", package_motion_script, # Use the full path
        "--input_dir", retargeted_dir,
        "--output_dir", packaged_dir
    ]
    run_command(package_cmd)

    print("\n--- Batch processing pipeline finished successfully! ---")

if __name__ == "__main__":
    main()