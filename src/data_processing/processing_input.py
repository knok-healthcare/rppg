import cv2
from os import walk, path, makedirs
from scipy.fft import fft, fftfreq
from skimage import io
import directories
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def convert_video_to_images(video_path, frame_rate):
    split_path = video_path.split('/')
    save_path = '/'.join(split_path[:-1]) + '/' + split_path[-1].replace('.mp4', '')
    if not path.exists(save_path):
        makedirs(save_path)
    clip = cv2.VideoCapture(video_path)
    success, image = clip.read()
    count = 0
    while success:
        cv2.imwrite(save_path + '/' + "frame{0}FR{1}.jpg".format(count, frame_rate), image)  # save frame as JPEG file
        success, image = clip.read()
        count += 1
    print('Images created.')


def process_finger_images(images_dir, frame_rate):

    period = 1/frame_rate

    filenames = next(walk(images_dir), (None, None, []))[2]
    averages = []
    for img_name in filenames:
        img = io.imread(images_dir + img_name, as_gray=False)
        mean_red = img[:, :, 0].mean()
        mean_green = img[:, :, 1].mean()
        mean_blue = img[:, :, 2].mean()
        averages.append([mean_red, mean_green, mean_blue])

    ts = pd.DataFrame(averages)  # time series
    ts.columns = ['red', 'green', 'blue']
    ts['red_ema'] = ts['red'].ewm(span=13).mean()  # Exponential moving average
    ts['green_ema'] = ts['green'].ewm(span=13).mean()
    ts['blue_ema'] = ts['green'].ewm(span=13).mean()
    ts.insert(0, 'time', ts.index * period)
    return ts


def plot_timeseries(ts):
    plt.plot(ts['time'], ts['red'], 'r--')
    plt.plot(ts['time'], ts['red_ema'], 'k.')
    plt.grid(visible=True)
    plt.show()

    plt.plot(ts['time'], ts['green'], 'g--')
    plt.plot(ts['time'], ts['green_ema'], 'k.')
    plt.show()

    plt.plot(ts['time'], ts['blue'], 'b--')
    plt.plot(ts['time'], ts['blue_ema'], 'k.')
    plt.show()
    print(1)


def get_fft(ts, frame_rate):
    # Number of sample points
    N = ts.shape[0]
    # sample spacing
    T = 1.0 / frame_rate
    xf = fftfreq(N, T)[:N // 2]

    y_red = ts['red']
    yf_red = fft(np.array(y_red))
    plt.plot(xf, 2.0 / N * np.abs(yf_red[0:N // 2]))
    plt.grid()
    plt.show()

    y_green = ts['green']
    yf_green = fft(np.array(y_green))
    plt.plot(xf, 2.0 / N * np.abs(yf_green[0:N // 2]))
    plt.grid()
    plt.show()

    y_blue = ts['blue']
    yf_blue = fft(np.array(y_blue))
    plt.plot(xf, 2.0 / N * np.abs(yf_blue[0:N // 2]))
    plt.grid()
    plt.show()


if __name__ == "__main__":
    directories.change_dir_to_main()
    frame_rate = 30
    # convert_video_to_images("data/examples/BPM81OX97FR30.mp4", frame_rate)
    ts = process_finger_images("data/examples/BPM68OX97FR30/", frame_rate)
    get_fft(ts, frame_rate)

