import directories
import numpy as np
import image_processing as ip
import pandas as pd
import signal_processing as sp


def prepare_fft_df(input_file, output_dir, frame_rate, parts):

    images = ip.convert_video_to_images(input_file)

    ts = ip.process_finger_video_in_parts(images, frame_rate=30, parts=parts)

    output_df = sp.get_rfft_for_models(ts, frame_rate, parts=parts)

    filename = input_file.split('/')[-1].replace('.mp4', '')
    output_df.reset_index().to_feather(output_dir + filename)


def read_fft_df(input_file, frame_rate, parts):
    readFrame = pd.read_feather(input_file, columns=None, use_threads=True).drop(columns=['index'])


if __name__ == "__main__":
    directories.change_dir_to_main()
    frame_rate = 30
    parts = 5
    input_file = 'data/examples/finger/BPM80T15.mp4'
    output_dir = 'data/prepared_model_input/fft/'
    prepare_fft_df(input_file, output_dir, frame_rate, parts)
    # read_fft_df(output_dir + 'BPM80T15', frame_rate, parts)