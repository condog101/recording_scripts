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
