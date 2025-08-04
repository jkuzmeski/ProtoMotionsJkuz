import typer
from pathlib import Path
import numpy as np

def main(
    input_file: Path = typer.Option(..., help="Path to the input text file containing motion data."),
    output_file: Path = typer.Option(..., help="Path to save the output .npy file.")
):
    """
    Converts a tab-delimited text file with motion data into a NumPy array.
    The script assumes the first 5 rows are headers and skips them.
    The data is expected to be structured as (frames, joints*3), and will be reshaped to (frames, joints, 3).
    """
    print(f"Loading data from {input_file}...")

    try:
        # Load the text file, skipping the header rows.
        data = np.loadtxt(input_file, skiprows=5)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        raise typer.Exit(code=1)

    # The first column is the frame number, so we skip it.
    positions_flat = data[:, 1:]
    
    # Check if the number of data columns is divisible by 3 (for X, Y, Z coordinates).
    if positions_flat.shape[1] % 3 != 0:
        print(f"Error: The number of data columns ({positions_flat.shape[1]}) is not a multiple of 3.")
        raise typer.Exit(code=1)
        
    num_frames = positions_flat.shape[0]
    num_joints = positions_flat.shape[1] // 3
    
    print(f"Found {num_frames} frames and {num_joints} joints.")
    
    # Reshape the data to (num_frames, num_joints, 3).
    positions_reshaped = positions_flat.reshape((num_frames, num_joints, 3))
    
    # Save the reshaped data to a .npy file.
    np.save(output_file, positions_reshaped)
    
    print(f"Successfully converted data and saved to {output_file}")
    print(f"Output array shape: {positions_reshaped.shape}")

if __name__ == "__main__":
    typer.run(main)
