import cv2
import cv2.ml as ml
import numpy as np
import os
from prnu import extract_multiple_aligned, extract_single

def train(dirs: list, labels: list, t: int, kernel: int, term_criteria: tuple):
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
    svm = ml.SVM_create()
    svm.setType(t)
    svm.setKernel(kernel)
    svm.setTermCritera(term_criteria)

    # get PRNU data for all images in directory
    prnus = list()
    prnu_labels = list()
    for directory, label in zip(dirs, labels):
        imgs = list()
        for fname in os.listdir():
            img = cv2.imread(os.path.join(directory, fname), cv2.IMREAD_COLOR)
            imgs.append(img)
            prnu_labels.append(label)
        prnu = extract_multiple_aligned(imgs)
        prnus.extend(prnu)

    svm.train(np.array(prnus), ml.ROW_SAMPLE, prnu_labels)
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
    labels = list()
    for fname in os.listdir(in_dir):
        img = cv2.imread(os.path.join(in_dir, fname))
        prnu = extract_single(img)
        labels.append(svm.classify(prnu))
    return int(np.average(labels).round())
