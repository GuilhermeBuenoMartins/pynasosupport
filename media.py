"""
Fuctions to treat media
"""

import cv2 as cv
import os

import numpy as np


def get_inputs(path='./'):
    """
    Get list of files from path

    :param path: path value
    :return: list of files
    """

    print('Reading dictory "{}"'.format(path))
    return os.listdir(path)


def read_video(file_name, path='./'):
    """
    Read a video in path

    :param file_name: video name
    :param path: path value
    :return: video
    """

    print('Reading "{}"...'.format(file_name))
    return cv.VideoCapture(os.path.join(path, file_name))


def save_image(image, file_name, path='./'):
    """
    Save image

    :param image: image
    :param file_name: image name
    :param path: path value
    """

    if not os.path.exists(path):
        os.makedirs(path)
    print('Saving the image "{}"...'.format(file_name))
    cv.imwrite(os.path.join(path, file_name), image)


def extract_frame(video_capture, video_name, fps=30, path='./'):
    """
    Extract frame from video

    :param video_capture: video
    :param video_name: video name
    :param fps: frames per seconds
    """

    print('Extracting frames...')
    video_name = video_name.split('.')[0]
    frame_path = path + '/{}'.format(video_name)
    frame_number = 0
    video_fps = video_capture.get(cv.CAP_PROP_FPS)
    fps = video_fps if fps > video_fps else fps
    print('video_fps={}'.format(video_fps))
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
    print('Extraction done.')
    video_capture.release()
    cv.destroyAllWindows()


def read_imgs(path='./', output_size=None) -> list:
    """
    The function read multiple images from indicated path redimensioning each output images.

    :param path: directory where images are.
    :param output_size: output size images.
    :return: list of numpy matrix
    """
    img_list = []
    if output_size is not None:
        print('Images output size = {}'.format(output_size))
        for file_name in get_inputs(path):
            print('Reading image {}...'.format(file_name))
            file = os.path.join(path, file_name)
            img = cv.cvtColor(cv.imread(file), cv.COLOR_BGR2RGB)
            img_list.append(cv.resize(img, dsize=output_size, interpolation=cv.INTER_CUBIC))
    else:
        for file_name in get_inputs(path):
            print('Reading image {}...'.format(file_name))
            file = os.path.join(path, file_name)
            img_list.append(cv.cvtColor(cv.imread(file), cv.COLOR_BGR2RGB))
    return img_list
