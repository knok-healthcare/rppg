from scipy.fft import fft, fftfreq
import directories
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal


def get_ema(ts):
    ts['red_ema'] = ts['red'].ewm(span=13).mean()  # Exponential moving average
    ts['green_ema'] = ts['green'].ewm(span=13).mean()
    ts['blue_ema'] = ts['blue'].ewm(span=13).mean()
    return ts


def plot_timeseries(ts, *cols):
    """ By default, prints all the colors, the cols is the bonus abbreviation"""

    for color in ['red', 'green', 'blue']:
        for column in cols:
            plt.grid(visible=True)
            plt.step(ts['time'], ts[color], 'r--')
            plt.step(ts['time'], ts[color + '_' + column], 'k--')
            plt.show()


def plot_fft(ts, frame_rate):
    # Number of sample points
    N = ts.shape[0]
    # sample spacing
    T = 1.0 / frame_rate
    xf = fftfreq(N, T)[:N // 2]

    y_red = ts['red_ema']
    yf_red = fft(np.array(y_red))
    plt.plot(xf, 2.0 / N * np.abs(yf_red[0:N // 2]))
    plt.grid()
    plt.show()

    y_green = ts['green_ema']
    yf_green = fft(np.array(y_green))
    plt.plot(xf, 2.0 / N * np.abs(yf_green[0:N // 2]))
    plt.grid()
    plt.show()

    y_blue = ts['blue_ema']
    yf_blue = fft(np.array(y_blue))
    plt.plot(xf, 2.0 / N * np.abs(yf_blue[0:N // 2]))
    plt.grid()
    plt.show()


def plot_ifft():
    pass


def plot_derivative(ts):
    ts['red_d'] = ts['red'].diff()/ts['time'].diff()
    ts['green_d'] = ts['green'].diff()/ts['time'].diff()
    ts['blue_d'] = ts['blue'].diff()/ts['time'].diff()
    plot_timeseries(ts, 'd')


def plot_periodogram(ts, frame_rate):
    f, Pxx_den = signal.periodogram(ts['red'], frame_rate, 'flattop', scaling='spectrum')
    plt.semilogy(f, Pxx_den)
    # plt.ylim([1e-7, 1e2])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()


def find_peaks():
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(x, height=0)


if __name__ == "__main__":
    directories.change_dir_to_main()

    # ts = process_finger_images("data/examples/BPM68OX97FR30/", frame_rate)
    # plot_fft(ts, frame_rate)
    # plot_derivative(ts)
    # plot_periodogram(ts, frame_rate)
