
import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A, PyK4ARecord, ImageFormat
from argparse import ArgumentParser
from typing import Optional, Tuple
import os
import datetime
import shortuuid


# Load camera with the default config

def generate_uid():
    return shortuuid.ShortUUID().random(length=8)

def get_file_name(uid):
    now = datetime.datetime.now()
    datestr = now.strftime("_%Y%m%d_%H%M%S")
    generated_name = uid + datestr + (".mkv")
    return generated_name

def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])  # type: ignore
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img

def init_record(device, outputfolder, uid, config):
    file_name = get_file_name(uid)
    outputpath = os.path.join(outputfolder, file_name)
    record = PyK4ARecord(device=device, config=config, path=outputpath)
    record.create()
    return record, outputpath

def main(outputfolder):
    if not os.path.isdir(outputfolder):
        raise ValueError(f"{outputfolder} is not a valid directory")
    uid = generate_uid()
    
    config = Config(
            color_format=ImageFormat.COLOR_MJPG,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    k4a = PyK4A(
        config
    )
    k4a.start()

   
    # Capture a single color image
    k4a.save_calibration_json("calibration.json")

    continue_flag = True
    record_flag = False
    show_rgb = True
    print("To start recording, press 's' key, to stop sequence press 's' key again. To terminate, press 'q'")
    print("Press x to toggle between depth and colour views")
    record, outputfile = init_record(k4a, outputfolder, uid, config)
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    cornerLoc = (0,25)
    fontScale              = 1
    fontColor              = (255,255,255)
    thickness              = 2
    lineType               = 2
    try:
        
        while True and continue_flag:

            capture = k4a.get_capture()
  
            if np.any(capture.color) and np.any(capture.depth):
                if show_rgb:
                    img = cv2.cvtColor(cv2.imdecode(capture.color, cv2.IMREAD_COLOR), cv2.COLOR_BGR2BGRA)
                else:
                    img =  colorize(capture.depth, (None, 5000), cv2.COLORMAP_HSV)
                if record_flag:
                    cv2.putText(img, 'Recording',
                                cornerLoc, 
                                font, 
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)
                    
                    record.write_capture(capture)
                if img is not None and img.shape[0] > 0:
                    cv2.imshow("k4a", img)
                    key = cv2.waitKey(10)
                if key == ord('x'):
                    show_rgb = not show_rgb
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    if record_flag:
                        print(f"Writing sequence to path {outputfile}")
                    break
            if key == ord('s'):
                record_flag = not record_flag
                if record_flag:
                    if record is None:
                        record, outputfile = init_record(k4a, outputfolder, uid, config)
                else:
                    print(f"Writing sequence to path {outputfile}")
                    record = None

            
  
           
    except KeyboardInterrupt:
        print("CTRL-C pressed. Exiting.")
    k4a.stop()
if __name__ == "__main__":
    parser = ArgumentParser(description = 'Args for record script')
    parser.add_argument('outputfolder', help='path_to_outputfolder')
    args = parser.parse_args()

    main(args.outputfolder)