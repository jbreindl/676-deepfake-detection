import cv2
import numpy as np
import numpy.typing
def extract(image_path : str, savepath : str) -> np.typing.NDArray[np.uint8]:
    """
    Fits the image to the face.
    Params:
        image_path - the image path to crop to face
        savepath - where to save the frame
    returns:
        cropped image
    """
    pass


def convert(path: str, frame_number: int) -> list:
    """
    Converts a video to its frames
    Params:
        path - path to the video
        frame_number - take every frame_number frame
    returns:
        returns list of frames of video
    """
    return_list = []
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        frame = 1
        ret, current_frame = cap.read()
        if not ret:
            cap.release()
            break
        if frame_number % frame  != 0:
            frame += 1
        else:
            frame += 1
            return_list.append(current_frame)
    print(return_list)

if __name__ == '__main__':
    convert('./aassnaulhq.mp4', 3)


    