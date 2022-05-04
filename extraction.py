import cv2
import os
from cv2 import imwrite
import numpy as np
def extract(image_path : str, savepath : str, frame_number = 3) -> None:
    """
    Fits the image to the face.
    Params:
        image_path - the image path to crop to face
        savepath - where to save the frame
        frame_number - take every frame_number frame
    returns:
        cropped image
    """
    list_of_images = convert(image_path, frame_number)
    frame = 0
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    video_name = image_path.split('/')[-1]
    cv2_dir = os.path.dirname(cv2.__file__)
    face_dir = os.path.join(cv2_dir, 'data/haarcascade_frontalface_alt.xml')
    haar = cv2.CascadeClassifier()
    haar.load(cv2.samples.findFile(face_dir))
    if list_of_images is None:
        return
    for current_image in list_of_images:
        img_grey = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        img_grey = cv2.equalizeHist(img_grey) # in opencv doc don't know if necessary
        faces = haar.detectMultiScale(img_grey)
        face_number = 0
        frame += 1
        for face in faces:
            x = face[0]
            y = face[1]
            w = face[2]
            h = face[3]
            current_face = current_image[y:y + h, x: x + w]
            current_face = cv2.resize(current_face, dsize = (128, 128))
            path = str(savepath) + '/' + video_name + '_' + str(frame) + '_' + str(face_number) + '.png'
            imwrite(path, current_face)
            face_number += 1
            print(frame)
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
            break
        if frame_number % frame != 0:
            frame += 1
        else:
            frame += 1
            return_list.append(current_frame)
    cap.release()
    return return_list

if __name__ == '__main__':
   # print(convert('./aassnaulhq.mp4', 3))
    extract('./aassnaulhq.mp4', './preprocessed', 3)


    