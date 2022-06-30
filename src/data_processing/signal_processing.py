import directories
from matplotlib import pyplot as plt
import numpy as np
import src.data_processing.image_processing as ip
import pandas as pd


def get_ema(ts):
    """ Exponential moving average. """
    red_ema = ts['red'].ewm(span=13).mean()  # Exponential moving average
    green_ema = ts['green'].ewm(span=13).mean()
    blue_ema = ts['blue'].ewm(span=13).mean()
    return red_ema, green_ema, blue_ema


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


def plot_multipart_timeseries(ts, *cols):
    """ By default, prints all the colors, the cols is the bonus abbreviation"""
    parts = len(ts.columns[ts.columns.str.contains('red')])
    for color in ['red', 'green', 'blue']:
        plt.grid(visible=True)

        for part in range(parts):
            if len(cols) > 0:
                for column in cols:
                    plt.step(ts['time'], ts[str(part) + '_' + color], 'r--')
                    plt.step(ts['time'], ts[str(part) + '_' + color + '_' + column], 'k--')
            else:
                plt.step(ts['time'], ts[str(part) + '_' + color], 'r--')
        plt.legend()
        plt.show()


def get_fft(ts, frame_rate, plot=True):
    from scipy.fftpack import fft, ifft, fftfreq

    max_heartrate = 180
    min_heartrate = 40

    # Number of sample points
    N = ts.shape[0]
    # sample spacing
    T = 1.0 / frame_rate
    W = fftfreq(N, d=T)

    # If our original signal time was in seconds, this is now in Hz
    f_signal = fft(np.array(ts['red']))
    cut_f_signal = f_signal.copy()
    cut_f_signal[(W < min_heartrate / 60) | (W > max_heartrate / 60)] = 0
    cut_signal_r = ifft(cut_f_signal)
    if plot:
        plt.plot(W, cut_f_signal)
        plt.grid()
        plt.show()

        plt.plot(ts['time'], cut_signal_r)
        plt.grid()
        plt.show()

    # If our original signal time was in seconds, this is now in Hz
    f_signal = fft(np.array(ts['green']))
    cut_f_signal = f_signal.copy()
    cut_f_signal[(W < min_heartrate / 60) | (W > max_heartrate / 60)] = 0
    # cut_f_signal[(W < min_heartrate/60)] = 0  # almost the same, except higher frequencies are not filtered out.
    cut_signal_g = ifft(cut_f_signal)
    if plot:
        plt.plot(W, cut_f_signal)
        plt.grid()
        plt.show()

        plt.plot(ts['time'], cut_signal_g)
        plt.grid()
        plt.show()

    # If our original signal time was in seconds, this is now in Hz
    f_signal = fft(np.array(ts['blue']))
    cut_f_signal = f_signal.copy()
    cut_f_signal[(W < min_heartrate / 60) | (W > max_heartrate / 60)] = 0
    cut_signal_b = ifft(cut_f_signal)
    if plot:
        plt.plot(W, cut_f_signal)
        plt.grid()
        plt.show()

        plt.plot(ts['time'], cut_signal_b)
        plt.grid()
        plt.show()

    return cut_signal_r, cut_signal_g, cut_signal_b


def get_rfft(ts, frame_rate, parts=1, plot=True):
    from scipy.fftpack import rfft, irfft, fftfreq

    max_heartrate = 500
    min_heartrate = 10

    # Number of sample points
    N = ts.shape[0]
    # sample spacing
    T = 1.0 / frame_rate
    W = fftfreq(N, d=T)
    W_disp = W[np.where(W == 0)[0][0]:np.where(W == 10)[0][0]]

    if parts == 1:
        output = {}
        for color in ['red', 'green', 'blue']:
            # If our original signal time was in seconds, this is now in Hz
            f_signal = rfft(np.array(ts[color]))
            cut_f_signal = f_signal.copy()
            # cut_f_signal[(W < min_heartrate / 60) | (W > max_heartrate / 60)] = 0
            output[color] = irfft(cut_f_signal)
            if plot:
                plt.plot(W, cut_f_signal)
                plt.grid()
                plt.show()

                plt.plot(ts['time'], output[color])
                plt.grid()
                plt.show()

        return output['red'], output['green'], output['blue']

    else:
        output = {}
        for color in ['red', 'green', 'blue']:
            output[color] = {}
            for part in range(parts):

                # If our original signal time was in seconds, this is now in Hz
                f_signal = rfft(np.array(ts[str(part) + '_' + color]))
                cut_f_signal = f_signal.copy()
                cut_f_signal[(W < min_heartrate / 60) | (W > max_heartrate / 60)] = 0

                output[color][str(part)] = irfft(cut_f_signal)
                if plot:
                    plt.plot(W, cut_f_signal)
                    plt.grid()
                    plt.show()

                    plt.plot(ts['time'], output[color][str(part)])
                    plt.plot(ts['time'], ts[str(part) + '_' + color])
                    plt.grid()
                    plt.show()

        return output['red'], output['green'], output['blue']


def get_rfft_for_models(ts, frame_rate, parts=1):
    from scipy.fftpack import rfft, fftfreq

    max_heartrate = 500
    min_heartrate = 10

    # Number of sample points
    N = ts.shape[0]
    # sample spacing
    T = 1.0 / frame_rate
    W = fftfreq(N, d=T)

    if parts == 1:
        output = {}
        for color in ['red', 'green', 'blue']:
            # If our original signal time was in seconds, this is now in Hz
            f_signal = rfft(np.array(ts[color]))
            cut_f_signal = f_signal.copy()
            cut_f_signal[(W < min_heartrate / 60) | (W > max_heartrate / 60)] = 0
            output[color] = cut_f_signal[np.where(W==0)[0][0]:np.where(W==10)[0][0]]

        return pd.DataFrame(output)

    else:
        output = {}
        for color in ['red', 'green', 'blue']:
            output[color] = {}
            for part in range(parts):

                # If our original signal time was in seconds, this is now in Hz
                f_signal = rfft(np.array(ts[str(part) + '_' + color]))
                cut_f_signal = f_signal.copy()
                cut_f_signal[(W < min_heartrate / 60) | (W > max_heartrate / 60)] = 0

                output[color][str(part)] = cut_f_signal[np.where(W==0)[0][0]:np.where(W==10)[0][0]]


        return pd.DataFrame(output)


def plot_derivative(ts):
    ts['red_d'] = ts['red'].diff() / ts['time'].diff()
    ts['green_d'] = ts['green'].diff() / ts['time'].diff()
    ts['blue_d'] = ts['blue'].diff() / ts['time'].diff()
    plot_timeseries(ts, 'd')


def plot_periodogram(ts, frame_rate):
    from scipy.signal import periodogram
    f, Pxx_den = periodogram(ts['red'], frame_rate, 'flattop', scaling='spectrum')
    plt.semilogy(f, Pxx_den)
    # plt.ylim([1e-7, 1e2])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()


def plot_spectrogram(ts, frame_rate):
    from scipy import signal
    import matplotlib.pyplot as plt
    from scipy.fft import fftshift

    f, t, Sxx = signal.spectrogram(np.array(ts['blue']), frame_rate, return_onesided=True)
    plt.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def find_local_maxima(ts):
    """ Returns positions of local maxima of sinusoid representing amount of blood in vessels. Should be used on strongly
    filtered signal like ifft. """
    from scipy.signal import argrelextrema
    return argrelextrema(ts, np.greater)


def cheby_filter(ts, frame_rate, parts=1):
    from scipy.signal import cheby1, sosfilt
    import matplotlib.pyplot as plt

    max_heartrate = 180
    min_heartrate = 40

    # Desired filter parameters
    order = 8 # adjusted experimentally, may need updates
    Apass = 2  # dB, adjusted experimentally, may need updates
    fmin = min_heartrate/60  # Hz
    fmax = max_heartrate/60  # Hz

    # Normalized frequency argument for cheby1
    # wn = fcut / (0.5 * fs)

    sos = cheby1(N=order,rp=Apass, Wn = [fmin, fmax], fs=frame_rate, btype='bandpass', output='sos')

    if parts == 1:
        for color in ['red', 'green', 'blue']:
            filtered = sosfilt(sos, ts[color])
            plt.plot(ts['time'], filtered)
            plt.plot(ts['time'], ts[color])

            plt.grid()
            plt.show()
    else:
        for color in ['red', 'green', 'blue']:
            for part in range(parts):
                filtered = sosfilt(sos, ts[str(part) + '_' + color])
                plt.plot(ts['time'], filtered)
                # plt.plot(ts['time'], ts[str(part) + '_' + color])

            plt.grid()
            plt.show()


def calculate_phase_shift(y1, y2):
    """ Attempts to calculate phase difference between two signals. May be useful in the future, especially if some
    frequency filtered signal would be used."""
    rdata = pd.concat([y1, y2], axis=1, ignore_index=True)
    from sklearn.preprocessing import StandardScaler
    import math
    scaler = StandardScaler()
    scaler.fit(rdata)
    data = scaler.transform(rdata)

    plt.figure()
    plt.plot(data[1:300, :])
    plt.show()

    # phase difference determination
    plt.figure(figsize=(4, 4))
    plt.title('Phase diagram')
    plt.scatter(data[1:100, 0], data[1:100, 1])
    plt.show()

    c = np.cov(np.transpose(data))
    print('cov: ', c)
    phi = np.arccos(c[0, 1])
    print('phase estimate (radians): ', phi, '(degrees): ', phi / math.pi * 180)


def plot_cwt_ricker(ts):
    """ Continuous wavelet transform. """
    from scipy import signal
    widths = np.arange(1, 31)

    for color in ['red', 'green', 'blue']:
        cwtmatr = signal.cwt(ts[color], signal.ricker, widths)
        plt.imshow(cwtmatr, extent=[-1, 1, 31, 1], cmap='PRGn', aspect='auto',
                   vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
        plt.show()


def plot_cwt_morlet2(ts, frame_rate):
    """ Wavelet transform with morlet wavelet. Generally outcome is not satisfying. Discontinued."""
    from scipy import signal
    import matplotlib.pyplot as plt
    w = 0.025
    freq = np.linspace(0.5, frame_rate / 10, 64)
    widths = w * frame_rate / (2 * freq * np.pi)

    for color in ['red', 'green', 'blue']:
        cwtm = signal.cwt(ts[color], signal.morlet2, widths, w=w)
        plt.pcolormesh(ts['time'], freq, np.abs(cwtm), cmap='viridis', shading='gouraud')
        plt.show()


if __name__ == "__main__":
    directories.change_dir_to_main()
    frame_rate = 30
    parts = 5


    images = ip.convert_video_to_images('data/examples/finger/BPM80T15.mp4')

    ts = ip.process_finger_video_in_parts(images, frame_rate=30, parts=parts)

    red, green, blue = get_rfft(ts, frame_rate, parts=parts, plot=True)



    # calculate_phase_shift(ts['0_blue'], ts['4_blue'])
    # plot_multipart_timeseries(ts)
    # cheby_filter(ts, frame_rate, parts=5)
    # ts = ip.process_finger_video(images, frame_rate=30)
    # plot_timeseries(ts)
    # plot_cwt_morlet2(ts, frame_rate)
    # ts_r, ts_g, ts_b = get_fft(ts, frame_rate, plot=True)
    #
    # get_rfft(ts, frame_rate)
    # plot_spectrogram(ts, frame_rate)
    # plot_periodogram(ts, frame_rate)
    print(1)
