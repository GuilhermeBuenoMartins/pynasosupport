"""
Fuctions to treat media
"""

import cv2 as cv
import logging as log
import os

import numpy as np


def get_inputs(path='input'):
    """
    Get list of files from path

    :param path: path value
    :return: list of files
    """

    log.info('Reading dictory "{}"'.format(path))
    return os.listdir(path)


def read_video(file_name, path='../input'):
    """
    Read a video in path

    :param file_name: video name
    :param path: path value
    :return: video
    """

    log.debug('Reading "{}"...'.format(file_name))
    return cv.VideoCapture(os.path.join(path, file_name))


def save_image(image, file_name, path='../output'):
    """
    Save image

    :param image: image
    :param file_name: image name
    :param path: path value
    """

    if not os.path.exists(path):
        os.makedirs(path)
    log.debug('Saving the image "{}"...'.format(file_name))
    cv.imwrite(os.path.join(path, file_name), image)


def extract_frame(video_capture, video_name, fps=30):
    """
    Extract frame from video

    :param video_capture: video
    :param video_name: video name
    :param fps: frames per seconds
    """

    log.debug('Extracting frames...')
    video_name = video_name.split('.')[0]
    frame_path = '../output/{}'.format(video_name)
    frame_number = 0
    video_fps = video_capture.get(cv.CAP_PROP_FPS)
    fps = video_fps if fps > video_fps else fps
    log.debug('video_fps={}'.format(video_fps))
    while video_capture.isOpened():
        exist, frame = video_capture.read()
        if not exist:
            break
        frame_number = frame_number + 1
        if frame_number % fps == 0:
            frame_name = '{}_frame_{:0>6}-second_{:0>3.0f}.jpg'.format(
                video_name, frame_number, frame_number // video_fps)
            cv.imshow('Frame', frame)
            save_image(frame, frame_name, frame_path)
    log.debug('Extraction done.')
    video_capture.release()
    cv.destroyAllWindows()


def read_imgs(path='input', factor=.0):
    """
    The function read multiples images from indicated path using a factor of reduction.

    :param path: directory where images are.
    :param factor: rescale factor of image.
    :return: list of numpy matrix
    """

    if factor > 0:
        log.debug('Reducing images with {} factor...'.format(factor))
        img_list = []
        for file_name in get_inputs(path):
            log.debug('Reading image {}...'.format(file_name))
            img = cv.cvtColor(cv.imread(os.path.join(path, file_name)), cv.COLOR_BGR2RGB)
            m = round(np.size(img, 0) * (1 - factor))
            n = round(np.size(img, 1) * (1 - factor))
            img_list.append(cv.resize(img, dsize=(m, n), interpolation=cv.INTER_CUBIC))
        return img_list
    log.error('Factor {} is invalid. Choose a number greater than 0.')
    return None


