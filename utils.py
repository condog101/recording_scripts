from pyk4a import ImageFormat, PyK4APlayback
import cv2
import webcolors


def create_bounding_box(center_x, center_y, box_size=10):
    half_size = box_size // 2

    top_left_x = center_x - half_size
    top_left_y = center_y - half_size

    bottom_right_x = center_x + half_size
    bottom_right_y = center_y + half_size

    return (top_left_x, top_left_y, bottom_right_x, bottom_right_y)


def process_time_string(keypoints: str):
    try:
        return [int(x.strip()) for x in keypoints.split(',')]
    except ValueError:
        print("Could not parse keypoints, returning empty list")
        return []


def is_closest_to_keypoints(keypoints: list, keypoint_dict: dict, time: float):
    for keypoint in keypoints:
        if keypoint not in keypoint_dict:
            int_time = int(time)
            if int_time == keypoint:
                return keypoint
    return None


def get_scene_image_from_capture(capture):
    capture._color = cv2.cvtColor(cv2.imdecode(
        capture.color, cv2.IMREAD_COLOR), cv2.COLOR_BGR2BGRA)
    capture._color_format = ImageFormat.COLOR_BGRA32

    colors = capture.color[..., (2, 1, 0)]
    return colors


def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")


def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def get_color_name(rgb_tuple):
    try:
        # Convert RGB to hex
        hex_value = webcolors.rgb_to_hex(rgb_tuple)
        # Get the color name directly
        return webcolors.hex_to_name(hex_value)
    except ValueError:
        # If exact match not found, find the closest color
        return closest_color(rgb_tuple)
