import cv2
import numpy as np
from pathlib import Path


def fetch_last_non_blank_frame(video_path, allowed_blank_frame_deviation=40):
    """
        Non blank frame in terms of this method is considered
        as frame which is not one color frame or at least one color frame
        Arguments:
            video_path: path to mp4 video file as string
            allowed_blank_frame_deviation: the higher parameter is, the bigger
                deviation of 'non empty frame' from one color frame is.
                example:
                    if set to 0 - only frames with the same
                        color of all pixels will be considered as blank
                    if set to 255 or more - all frames will be considered as blank
        Returns:
            path to .png file that contains 'non blank frame'
            if such exists, else None
    """
    def is_blank(frame_array):
        """
        Judges if frame_array is blank.
        Treats given frame_array as 3 dimensional vector space of pixels.
        Counts 'gravity center' of pixels set and checks
        if every pixel of frame has allowed deviation of center.
        In the other words just checks if every pixel has
        at least the same color as 'gravity center' pixel.
        """
        size_x, size_y, pixel_size = frame_array.shape
        pixels_num = size_x * size_y
        gravity_center = np.sum(np.sum(frame_array, axis=0), axis=0) / pixels_num
        comparator_array = np.full((size_x, size_y, pixel_size), gravity_center)
        return (np.absolute(frame_array - comparator_array) <= allowed_blank_frame_deviation).all()

    result = None
    frames_stream = cv2.VideoCapture(video_path)

    ret, frame = frames_stream.read()
    while ret:
        if not is_blank(frame):
            result = frame
        ret, frame = frames_stream.read()
    frames_stream.release()

    if result is None:
        return None

    tmp_dir = Path('tmp')
    if not tmp_dir.exists():
        tmp_dir.mkdir()

    video_name, extension = Path(video_path).name.split('.')
    result_path = tmp_dir / f'{video_name}_non_blank_frame.png'
    cv2.imwrite(str(result_path), result)

    return result_path
