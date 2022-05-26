import cv2
import logging
import numpy as np
import os
"""
Functions to treat media
"""

def get_inputs(path='input'):
    """
    Get list of files from path
    :param path: path value
    :return: list of files
    """
    logging.info('Reading dictory "{}"'.format(path))
    return os.listdir(path)


def read_img(file_name, path='input'):
    """
    Read image from path
    :param file_name: image name
    :param path: path value
    :return: RGB image
    """
    logging.info('Reading "{}"...'.format(file_name))
    return cv2.cvtColor(cv2.imread(os.path.join(path, file_name)), cv2.COLOR_BGR2RGB)


def read_n_imgs(path='input', reduction_prcnt=0.0):
    """
    Read multiples images form path using a reduction factor of resolution.
    :param file_name: image name
    :param path: path value
    :param reduction_prcnt: reduction percent between 0.0 and 100.0 (exclusive)
    :return: list of images
    """
    if (0.0 <= reduction_prcnt) and (reduction_prcnt < 100.0):
        fctor = 1 - (reduction_prcnt / 100)
        imgs = []
        for file_name in get_inputs(path):
            img = read_img(file_name, path)
            h = round(np.size(img, 0) * fctor)
            w = round(np.size(img, 1) * fctor)
            imgs.append(cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_CUBIC))
        return imgs
    else:
        logging.warning('Value reduction_prcnt = {} incompatible to interval'.format(reduction_prcnt))
        return None


def read_video(file_name, path='../input'):
    """
    Read a video in path
    :param file_name: video name
    :param path: path value
    :return: video
    """
    logging.info('Reading "{}"...'.format(file_name))
    return cv2.VideoCapture(os.path.join(path, file_name))


def save_image(image, file_name, path='../output'):
    """
    Save image
    :param image: image
    :param file_name: image name
    :param path: path value
    """
    if not os.path.exists(path):
        os.makedirs(path)
    logging.info('Saving the image "{}"...'.format(file_name))
    cv2.imwrite(os.path.join(path, file_name), image)


def extract_frame(video_capture, video_name, fps=30):
    """
    Extract frame from video
    :param video_capture: video
    :param video_name: video name
    :param fps: frames per seconds
    """
    logging.info('Extracting frames...')
    video_name = video_name.split('.')[0]
    frame_path = '../output/{}'.format(video_name)
    frame_number = 0
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    fps = video_fps if fps > video_fps else fps
    logging.info("video_fps={}".format(video_fps))
    while video_capture.isOpened():
        exist, frame = video_capture.read()
        if not exist:
            break
        frame_number = frame_number + 1
        if frame_number % fps == 0:
            frame_name = '{}_frame_{:0>6}-second_{:0>3.0f}.jpg'.format(video_name, frame_number, frame_number//video_fps)
            cv2.imshow('Frame', frame)
            save_image(frame, frame_name, frame_path)
    logging.info('Extraction done.')
    video_capture.release()
    cv2.destroyAllWindows()