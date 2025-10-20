import os
import sys
import glob
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2  

def find_global_bounds(csv_files):
    """
    Scans all CSV files to find the absolute min/max for
    X, Y, Z, and absv. This is crucial for a stable video.
    """
    print("Scanning CSV files to determine global plot bounds...")
    g_min_x, g_max_x = np.inf, -np.inf
    g_min_y, g_max_y = np.inf, -np.inf
    g_min_z, g_max_z = np.inf, -np.inf
    g_min_v, g_max_v = np.inf, -np.inf

    for f in csv_files:
        # Read only necessary columns for speed
        try:    # Gracefully skip to the next frame if a CSV was saved erroneously
            df = pd.read_csv(f, usecols=['X', 'Y', 'Z', 'absv'])
        except:
            break        
        g_min_x = min(g_min_x, df['X'].min())
        g_max_x = max(g_max_x, df['X'].max())
        g_min_y = min(g_min_y, df['Y'].min())
        g_max_y = max(g_max_y, df['Y'].max())
        g_min_z = min(g_min_z, df['Z'].min())
        g_max_z = max(g_max_z, df['Z'].max())
        g_min_v = min(g_min_v, df['absv'].min())
        g_max_v = max(g_max_v, df['absv'].max())

    # Determine the maximum range to make all axes equal
    max_range = max(
        g_max_x - g_min_x,
        g_max_y - g_min_y,
        g_max_z - g_min_z
    )
    
    # Calculate the center point for each axis
    mid_x = (g_max_x + g_min_x) / 2
    mid_y = (g_max_y + g_min_y) / 2
    mid_z = (g_max_z + g_min_z) / 2

    # Return a dictionary of plot limits
    plot_limits = {
        'xlim': (mid_x - max_range / 2, mid_x + max_range / 2),
        'ylim': (mid_y - max_range / 2, mid_y + max_range / 2),
        'zlim': (mid_z - max_range / 2, mid_z + max_range / 2),
        'vmin': g_min_v,
        'vmax': g_max_v
    }
    print("Global bounds determined.")
    return plot_limits

def generate_frame(csv_filepath, output_image_path, plot_limits, file_index):
    """
    Generates a single 3D scatter plot image for a given CSV file
    using the pre-calculated global bounds.
    """
    df = pd.read_csv(csv_filepath)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Optional: Scale radius for visibility
    # sizes = df['r'] * 20000
    sizes = df['r'] 

    scatter = ax.scatter(
        df['X'],
        df['Y'],
        df['Z'],
        s=sizes,
        c=df['absv'],
        cmap='viridis',
        alpha=0.6,
        vmin=plot_limits['vmin'],  # Use global min for color
        vmax=plot_limits['vmax']   # Use global max for color
    )

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Add a title that shows the frame number
    ax.set_title(f'DEM Spheres (Frame: {file_index})', pad=20)

    cbar = fig.colorbar(scatter, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Absolute Velocity (absv)')

    # Apply the global limits for a stable camera
    ax.set_xlim(plot_limits['xlim'])
    ax.set_ylim(plot_limits['ylim'])
    ax.set_zlim(plot_limits['zlim'])

    plt.savefig(output_image_path, dpi=120, bbox_inches='tight')
    plt.close(fig)  # Close the figure to save memory

def compile_frames_to_video(frame_files, output_video_path, fps):
    """
    Stitches a list of frame image files into a video.
    """
    print(f"Compiling video at {fps} FPS...")
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    height, width, layers = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        img = cv2.imread(frame_file)
        video.write(img)

    video.release()
    cv2.destroyAllWindows()
    print(f"Video saved successfully to {output_video_path}")

def create_dem_video(input_folder, output_video="dem_simulation.mp4", fps=10):
    """
    Main function to create a video from a folder of DEM CSVs.
    """
    # 1. Define a temporary directory for frames
    temp_dir = "temp_frames"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)  # Clear old frames
    os.makedirs(temp_dir)

    # 2. Find and sort all CSV files
    search_pattern = os.path.join(input_folder, "DEMdemo_output_*.csv")
    csv_files = sorted(glob.glob(search_pattern))
    
    if not csv_files:
        print(f"Error: No files found matching '{search_pattern}'")
        return

    print(f"Found {len(csv_files)} CSV files to process.")

    # 3. Find global bounds for consistent plotting
    plot_limits = find_global_bounds(csv_files)

    # 4. Generate all frames
    frame_files = []
    for i, csv_file in enumerate(csv_files):
        print(f"  Generating frame {i+1}/{len(csv_files)}...")
        # Use zero-padding for correct file ordering
        frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
        generate_frame(csv_file, frame_path, plot_limits, i)
        frame_files.append(frame_path)

    # 5. Compile frames into a video
    compile_frames_to_video(frame_files, output_video, fps)

    # 6. Clean up temporary files
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    # Usage: python component_sphere_display.py <INPUT_CSV_FOLDER> [FPS]
    # e.g. python component_sphere_display.py ../../build/bin/DemoOutput_WheelSlopeSlip 30
    
    if len(sys.argv) < 2:
        print("Usage: python component_sphere_display.py <INPUT_CSV_FOLDER> [FPS]")
        sys.exit(1)

    # Resolve input folder (supports ~ and relative paths)
    INPUT_CSV_FOLDER = os.path.abspath(os.path.expanduser(sys.argv[1]))

    # FPS optional arg, default to 30
    OUTPUT_FPS = int(sys.argv[2]) if len(sys.argv) >= 3 else 30

    # Derive output video filename from folder name portion after the first underscore
    base_name = os.path.basename(os.path.normpath(INPUT_CSV_FOLDER))
    suffix = base_name.split('_', 1)[1] if '_' in base_name else base_name
    OUTPUT_VIDEO_FILE = f"{suffix}_{OUTPUT_FPS}fps.mp4"

    create_dem_video(INPUT_CSV_FOLDER, OUTPUT_VIDEO_FILE, OUTPUT_FPS)