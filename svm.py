import cv2
import cv2.ml as ml
import numpy as np
from prnu import extract_multiple_aligned

def train(dirs: list, labels: list, t: int, kernel: int):
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
    svm =  ml.SVM_create()
    svm.setType(t)
    svm.setKernel(kernel)
    return svm

def classify(dir: list, svm) -> int:
    """
    Classify an image using a trained SVM
    Params:
        dir: Directories corresponding to frames of a video
        svm: trained svm
    Returns:
        label: returns label of the data
    """
    return -1
