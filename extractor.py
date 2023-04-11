"""
Script to extract frames from videos
"""

import media

path = input('Enter project path: ')
input_path = path + '/input/'
output_path = path + '/output'
for video_name in media.get_inputs(path=input_path):
    video_capture = media.read_video(video_name, input_path)
    frames = media.extract_frame(video_capture, video_name, 5, path=output_path)
