import directories
from matplotlib import pyplot as plt
import numpy as np
import image_processing as ip


def get_ema(ts):
    ts['red_ema'] = ts['red'].ewm(span=13).mean()  # Exponential moving average
    ts['green_ema'] = ts['green'].ewm(span=13).mean()
    ts['blue_ema'] = ts['blue'].ewm(span=13).mean()
    return ts


def plot_timeseries(ts, *cols):
    """ By default, prints all the colors, the cols is the bonus abbreviation"""

    for color in ['red', 'green', 'blue']:
        if len(cols) > 0:
            for column in cols:
                plt.grid(visible=True)
                plt.step(ts['time'], ts[color], 'r--')
                plt.step(ts['time'], ts[color + '_' + column], 'k--')
                plt.show()
        else:
            plt.grid(visible=True)
            plt.step(ts['time'], ts[color], 'r--')
            plt.show()


def get_fft(ts, frame_rate, plot=True):
    from scipy.fft import fft, fftfreq

    # Number of sample points
    N = ts.shape[0]
    # sample spacing
    T = 1.0 / frame_rate
    xf = fftfreq(signal.size, d=time[1] - time[0])

    y_red = ts['red']
    fft_red = fft(np.array(y_red))
    if plot:
        plt.plot(xf, 2.0 / N * np.abs(fft_red[0:N // 2]))
        plt.grid()
        plt.show()

    y_green = ts['green']
    fft_green = fft(np.array(y_green))
    if plot:
        plt.plot(xf, 2.0 / N * np.abs(fft_green[0:N // 2]))
        plt.grid()
        plt.show()

    y_blue = ts['blue']
    fft_blue = fft(np.array(y_blue))
    if plot:
        plt.plot(xf, 2.0 / N * np.abs(fft_blue[0:N // 2]))
        plt.grid()
        plt.show()

    return fft_red, fft_green, fft_blue


def filter_fft(ts_fft, freq_bottom, freq_top):
    return ts_fft


def get_ifft(fft_df, filter_bot=0, filter_top=10000, plot=True):
    pass


def get_rfft(ts, frame_rate, plot=True):
    from scipy.fftpack import rfft, irfft, fftfreq

    max_heartrate = 180
    min_heartrate = 40

    # Number of sample points
    N = ts.shape[0]
    # sample spacing
    T = 1.0 / frame_rate
    W = fftfreq(N, d=T)

    # If our original signal time was in seconds, this is now in Hz
    f_signal = rfft(np.array(ts['red']))
    cut_f_signal = f_signal.copy()
    cut_f_signal[(W < min_heartrate / 60) | (W > max_heartrate / 60)] = 0
    cut_signal_r = irfft(cut_f_signal)
    if plot:
        plt.plot(W, cut_f_signal)
        plt.grid()
        plt.show()

        plt.plot(ts['time'], cut_signal_r)
        plt.grid()
        plt.show()

    # If our original signal time was in seconds, this is now in Hz
    f_signal = rfft(np.array(ts['green']))
    cut_f_signal = f_signal.copy()
    cut_f_signal[(W < min_heartrate / 60) | (W > max_heartrate / 60)] = 0
    # cut_f_signal[(W < min_heartrate/60)] = 0  # almost the same, except higher frequencies are not filtered out.
    cut_signal_g = irfft(cut_f_signal)
    if plot:
        plt.plot(W, cut_f_signal)
        plt.grid()
        plt.show()

        plt.plot(ts['time'], cut_signal_g)
        plt.grid()
        plt.show()

    # If our original signal time was in seconds, this is now in Hz
    f_signal = rfft(np.array(ts['blue']))
    cut_f_signal = f_signal.copy()
    cut_f_signal[(W < min_heartrate / 60) | (W > max_heartrate / 60)] = 0
    cut_signal_b = irfft(cut_f_signal)
    if plot:
        plt.plot(W, cut_f_signal)
        plt.grid()
        plt.show()

        plt.plot(ts['time'], cut_signal_b)
        plt.grid()
        plt.show()

    return cut_signal_r, cut_signal_g, cut_signal_b


def plot_derivative(ts):
    ts['red_d'] = ts['red'].diff() / ts['time'].diff()
    ts['green_d'] = ts['green'].diff() / ts['time'].diff()
    ts['blue_d'] = ts['blue'].diff() / ts['time'].diff()
    plot_timeseries(ts, 'd')


def plot_periodogram(ts, frame_rate):
    f, Pxx_den = signal.periodogram(ts['red'], frame_rate, 'flattop', scaling='spectrum')
    plt.semilogy(f, Pxx_den)
    # plt.ylim([1e-7, 1e2])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()


def find_local_maxima(ts):
    """ Returns positions of local maxima of sinusoid representing amount of blood in vessels. Should be used on strongly
    filtered signal like ifft. """
    from scipy.signal import argrelextrema
    return argrelextrema(ts, np.greater)


if __name__ == "__main__":
    directories.change_dir_to_main()
    frame_rate = 30
    images = ip.convert_video_to_images('data/examples/finger/BPM70OX97FR30.mp4')
    ts = ip.process_finger_video(images, frame_rate)
    # plot_timeseries(ts)

    ts_r, ts_g, ts_b = get_rfft(ts, frame_rate, plot=False)
    maxima = find_local_maxima(ts_r)[0]

    # Cuts signal to start and end with local maxima, so there are only entire periods taken into further calculations.
    time = ts['time'][maxima[-2]] - ts['time'][maxima[1]]

    heart_rate = ((len(maxima)-3)/time) * 60

    print(1)


    # plot_derivative(ts)
    # plot_periodogram(ts, frame_rate)
