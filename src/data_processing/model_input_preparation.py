import directories
import numpy as np
import src.data_processing.image_processing as ip
import pandas as pd
import src.data_processing.signal_processing as sp


def prepare_fft_df(input_file, output_dir, frame_rate, parts):

    images = ip.convert_video_to_images(input_file)

    ts = ip.process_finger_video_in_parts(images, frame_rate=30, parts=parts)

    output_df = sp.get_rfft_for_models(ts, frame_rate, parts=parts)

    filename = input_file.split('/')[-1].replace('.mp4', '')
    output_df.reset_index().to_feather(output_dir + filename)


def read_fft_df(input_file):
    df = pd.read_feather(input_file, columns=None, use_threads=True).drop(columns=['index'])
    arr = convert_fft_df_to_numpy(df)

    return arr


def convert_fft_df_to_numpy(df):
    arrays = []
    for col in df:
        arrays.append(np.array(df[col].tolist()))
    arr = np.dstack(arrays)

    return arr


if __name__ == "__main__":
    directories.change_dir_to_main()
    frame_rate = 30
    parts = 5
    input_file = 'data/examples/finger/BPM80T15.mp4'
    output_dir = 'data/prepared_model_input/fft/'
    # prepare_fft_df(input_file, output_dir, frame_rate, parts)
    df = read_fft_df(output_dir + 'BPM80T15')
    arr = convert_fft_df_to_numpy(df)
