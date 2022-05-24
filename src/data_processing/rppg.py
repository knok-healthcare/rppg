import signal_processing as sp
import image_processing as ip
import directories


def get_heart_rate(_video_dir, frame_rate):
    images = ip.convert_video_to_images(_video_dir)
    _ts = ip.process_finger_video(images, frame_rate)
    # plot_timeseries(ts)

    ts_r, ts_g, ts_b = sp.get_rfft(_ts, frame_rate, plot=False)
    maxima = sp.find_local_maxima(ts_r)[0]

    # Cuts signal to start and end with local maxima, so there are only entire periods taken into further calculations.
    time = _ts['time'][maxima[-1]] - _ts['time'][maxima[0]]

    # Full cycles per time times seconds in minute
    heart_rate = ((len(maxima)-1)/time) * 60

    return heart_rate



if __name__ == "__main__":
    directories.change_dir_to_main()

    frame_rate = 30
    video_dir = 'data/examples/finger/BPM70OX97FR30.mp4'
    print(get_heart_rate(video_dir, frame_rate))
