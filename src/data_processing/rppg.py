import signal_processing as sp
import image_processing as ip
import directories


if __name__ == "__main__":
    directories.change_dir_to_main()

    frame_rate = 30
    images = ip.convert_video_to_images('data/examples/face/BPM72OX97FR30.mp4')
    ts_left, ts_right = ip.process_face_video(images, 30)

    ts_left = sp.get_ema(ts_left)
    ts_right = sp.get_ema(ts_right)

    sp.plot_timeseries(ts_left)
    sp.plot_timeseries(ts_right)
