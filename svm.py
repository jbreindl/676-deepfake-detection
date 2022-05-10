import cv2
import numpy as np
import os
from sklearn.svm import SVR
from prnu import extract_multiple_aligned

def train(dirs: list, labels: list):
    """
    Train an SVM with the given parameters
    Params:
        dirs: List of directories corresponding to frames of a video
        labels: labels corresponding to the dirs
        t: type of svm to train (from cv2.ml)
        kernel: type of kernel to use (from cv2.ml)
    Returns:
        svm: trained svm
    """
    # instantiate svm
    svm = SVR()

    # get PRNU data for all images in directory
    prnus = list()
    for directory in dirs:
        imgs = list()
        for fname in os.listdir(directory):
            img = cv2.imread(os.path.join(directory, fname), cv2.IMREAD_COLOR)
            imgs.append(img)
        # get what is essentially PRNU data for whole video and prepare to classify
        prnu = extract_multiple_aligned(imgs)
        prnus.append(prnu.flatten())
    prnus = np.array(prnus)
    svm.fit(prnus, labels)
    return svm

def classify(in_dir: str, svm) -> int:
    """
    Classify an image using a trained SVM
    Params:
        dir: Directories corresponding to frames of a video
        svm: trained svm
    Returns:
        label: returns label of the data
    """
    images = list()
    for fname in os.listdir(in_dir):
        img = cv2.imread(os.path.join(in_dir, fname))
        images.append(img)
    prnus = extract_multiple_aligned(images)
    prediction = svm.predict(prnus.reshape(1, -1))
    return prediction

if __name__ == "__main__":
    trained = train(['preprocessed'], [1])
    print(classify('preprocessed', trained))
