from pyk4a import PyK4APlayback, ImageFormat
import cv2
import open3d as o3d
import pickle as pkl
import os
from tqdm import tqdm
import numpy as np

VIDEO_FOLDER = "/home/connorscomputer/Documents/CT_DATA/VIDEOS/"
EXTRACTED_POINTS_FOLDER = "/home/connorscomputer/Documents/CT_DATA/RESULTS/"
DUMP_POINTS_FOLDER = "/home/connorscomputer/Documents/CT_DATA/COMBINED_SEGMENT_CLOUDS/"


def extract_targets(videofilepath, targetfilepath, dump_folder):
    playback = PyK4APlayback(videofilepath)
    img = None
    pcd = None
    calibration = None
    transformed_depth = None
    index = 0
    # points_colours_targets = []
    # with open(targetfilepath, 'rb') as handle:
    #     targets = pkl.load(handle)

    os.makedirs(dump_folder, exist_ok=True)

    try:
        playback.open()
        while True:
            try:
                capture = playback.get_next_capture()

                if capture.color is not None and capture.depth is not None:

                    # target = targets[index][0]

                    capture._color = cv2.cvtColor(cv2.imdecode(
                        capture.color, cv2.IMREAD_COLOR), cv2.COLOR_BGR2BGRA)
                    capture._color_format = ImageFormat.COLOR_BGRA32

                    points = capture.transformed_depth_point_cloud.reshape(
                        (-1, 3)).astype('float64')

                    colors = capture.color[..., (2, 1, 0)].reshape(
                        (-1, 3))

                    if len(points) < 250:
                        index += 1
                        continue

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(
                        points)
                    pcd.colors = o3d.utility.Vector3dVector(
                        (colors/255).astype('float64'))
                    # calibration = capture._calibration
                    # transformed_depth = capture.transformed_depth
                    frame_name = f"frame_{index}.pcd"
                    dump_path = os.path.join(
                        dump_folder, frame_name)

                    o3d.io.write_point_cloud(dump_path, pcd, compressed=True)

                    index += 1

            except EOFError:
                break
    finally:
        playback.close()
    return None


def process_raw_data():
    """
    Process raw point cloud data into format needed for training

    Args:
        raw_point_clouds: List of (N, 3) arrays or single (B, N, 3) array
        raw_colors: Optional list of (N, 3) arrays or (B, N, 3) array with RGB values
    """

    for filename in tqdm(os.listdir(VIDEO_FOLDER)):
        file_path = os.path.join(VIDEO_FOLDER, filename)
        if os.path.isfile(file_path):    # Check if it's a file (not a directory)
            print(f"Processing file: {filename}")
            target_filename = "closest_points_"+filename.split(".")[0]+".pkl"
            target_path = os.path.join(
                EXTRACTED_POINTS_FOLDER, target_filename)

            dump_name = "source_points_"+filename.split(".")[0]
            dump_path = os.path.join(
                DUMP_POINTS_FOLDER, dump_name)
            if not os.path.exists(dump_path):
                extract_targets(file_path, target_path, dump_path)

    return None


process_raw_data()
# points_list = [
#     np.random.randn(np.random.randint(400, 600), 3)
#     for _ in range(num_clouds)
# ]
# colors_list = [
#     np.random.rand(points.shape[0], 3)
#     for points in points_list
# ]

# train_loader = create_dataloader(
#     points_list=points_list,
#     colors_list=colors_list,
#     batch_size=32,
#     target_points=2500
# )
