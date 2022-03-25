import os
import cv2


def get_inputs(path='../input'):
    print('Reading dictory "{}"'.format(path))
    return os.listdir(path)

def read_video(file_name, path='../input'):
    print('Reading "{}"...'.format(file_name))
    return cv2.VideoCapture(os.path.join(path, file_name))


def save_image(image, file_name, path='../output'):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Saving the image "{}"...'.format(file_name))
    cv2.imwrite(os.path.join(path, file_name), image)


def extract_frame(video_capture, video_name):
    print('Extracting frames...')
    frame_path = '../output/{}'.format(video_name.split('.')[0])
    frame_number = 0
    while video_capture.isOpened():
        exist, frame = video_capture.read()
        if exist:
            frame_number = frame_number + 1
            frame_name = 'frame_{:0>7}.jpg'.format(frame_number)
            cv2.imshow('Frame', frame)
            save_image(frame, frame_name, frame_path)
        else:
            break
    print('Extraction done.')
    video_capture.release()
    cv2.destroyAllWindows()