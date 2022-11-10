"""
Script to extract frames from videos
"""
import src.functions.media
from src.functions import media

for video_name in media.get_inputs():
    video_capture = media.read_video(video_name)
    frames = media.extract_frame(video_capture, video_name, 5)
