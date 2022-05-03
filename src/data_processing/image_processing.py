import dlib, cv2
import numpy as np
import directories
import pandas as pd
from skimage import io
from os import walk, path, makedirs

# Landmark's numbers according to dlib's 68 model
roi_landmarks = {'left_eye_left': 36,
                 'left_eye_right': 39,
                 'right_eye_left': 42,
                 'right_eye_right': 45,
                 'nose_mid': 29,
                 'nose_tip': 33}


def save_video_as_images(video_path, frame_rate):
    split_path = video_path.split('/')
    save_path = '/'.join(split_path[:-1]) + '/' + split_path[-1].replace('.mp4', '')
    if not path.exists(save_path):
        makedirs(save_path)
    clip = cv2.VideoCapture(video_path)
    success, img = clip.read()
    count = 0
    while success:
        cv2.imwrite(save_path + '/' + "frame{0}FR{1}.jpg".format(count, frame_rate), img)  # save frame as JPEG file
        success, img = clip.read()
        count += 1
    print('Images created.')


def convert_video_to_images(video_path):
    clip = cv2.VideoCapture(video_path)
    success, img = clip.read()
    images = []
    while success:
        images.append(img)
        success, img = clip.read()
    return images


def process_finger_video(images, frame_rate):
    period = 1 / frame_rate

    averages = []
    for img in images:
        mean_rgb = get_mean_rgb(img)
        averages.append([mean_rgb[0], mean_rgb[1], mean_rgb[2]])

    ts = pd.DataFrame(averages)  # time series
    ts.columns = ['red', 'green', 'blue']
    ts['red_ema'] = ts['red'].ewm(span=13).mean()  # Exponential moving average
    ts['green_ema'] = ts['green'].ewm(span=13).mean()
    ts['blue_ema'] = ts['green'].ewm(span=13).mean()
    ts.insert(0, 'time', ts.index * period)
    return ts


def process_face_video(images, frame_rate):
    """ Takes face mp4, finds ROIs (regions of interest), takes average RGB colors from each ROI, and returns time
    oriented data frames for further processing."""

    model_dir = directories.face_68_model_dir
    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor(model_dir)

    period = 1 / frame_rate

    averages_left_roi = []
    averages_right_roi = []
    for img in images:
        roi_left, roi_right = get_roi_images(face_detector, landmark_detector, img)

        cv2.imshow("Face landmark result", roi_left)
        cv2.imshow("Face landmark result", roi_right)
        mean_rgb_left = get_mean_rgb(roi_left)
        mean_rgb_right = get_mean_rgb(roi_right)

        averages_left_roi.append([mean_rgb_left[0], mean_rgb_left[1], mean_rgb_left[2]])
        averages_right_roi.append([mean_rgb_right[0], mean_rgb_right[1], mean_rgb_right[2]])

    ts_left = pd.DataFrame(averages_left_roi)  # time series
    ts_left.columns = ['red', 'green', 'blue']
    ts_left.insert(0, 'time', ts_left.index * period)

    ts_right = pd.DataFrame(averages_right_roi)  # time series
    ts_right.columns = ['red', 'green', 'blue']
    ts_right.insert(0, 'time', ts_right.index * period)

    return ts_left, ts_right


def get_mean_rgb(img):
    mean_red = img[:, :, 0].mean()
    mean_green = img[:, :, 1].mean()
    mean_blue = img[:, :, 2].mean()

    return mean_red, mean_green, mean_blue


def get_roi_coordinates(landmarks) -> dict:
    """ Gets necessary coordinates of points on face which will serve as boundaries. """

    return {'top': landmarks.part(roi_landmarks['nose_mid']).y,
            'bottom': landmarks.part(roi_landmarks['nose_tip']).y,
            'left_1': landmarks.part(roi_landmarks['left_eye_left']).x,
            'left_2': landmarks.part(roi_landmarks['left_eye_right']).x,
            'right_1': landmarks.part(roi_landmarks['right_eye_left']).x,
            'right_2': landmarks.part(roi_landmarks['right_eye_right']).x}


def draw_points(image, face_landmarks):
    """ Draws landmarks on the face. Useful for debug and presentation. """
    for i in range(len(face_landmarks.parts())):
        if i in roi_landmarks.values():
            point = [face_landmarks.part(i).x, face_landmarks.part(i).y]
            cv2.circle(image, point, radius=4, color=(0, 0, 255), thickness=-1)


def get_roi_images(face_detector, landmark_detector, image):
    """ Finds a face and landmarks. Takes necessary parts and returns concatenated."""

    # Now this line will try to detect all faces in an image either 1 or 2 or more faces
    face = face_detector(image, 0)[0]
    face_rectangle = dlib.rectangle(int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
    detected_landmarks = landmark_detector(image, face_rectangle)
    draw_points(image, detected_landmarks)
    cv2.imshow("Face landmark result", image)

    roi_coords = get_roi_coordinates(detected_landmarks)
    left_roi = image[roi_coords['left_1']:roi_coords['left_2'],
               roi_coords['top']:roi_coords['bottom']]  # todo this does not work properly yet
    right_roi = image[roi_coords['right_1']:roi_coords['right_2'], roi_coords['top']:roi_coords['bottom']]

    return left_roi, right_roi


if __name__ == "__main__":
    directories.change_dir_to_main()

    frame_rate = 30
    process_face_video(convert_video_to_images('data/examples/face/BPM72OX97FR30.mp4'), 30)

