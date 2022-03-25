import src.midia as midia


for video_name in midia.get_inputs():
    video_capture = midia.read_video(video_name)
    frames = midia.extract_frame(video_capture, video_name)
