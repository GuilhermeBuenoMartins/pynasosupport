"""
Script to extract frames from videos
"""
import src.media as media


for video_name in media.get_inputs():
    video_capture = media.read_video(video_name)
    frames = media.extract_frame(video_capture, video_name, 5)
