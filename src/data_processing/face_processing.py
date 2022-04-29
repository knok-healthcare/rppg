import dlib, cv2
import numpy as np
import directories
import image_signal_processing
# Landmark's numbers according to dlib's 68 model
roi_landmarks = {'left_eye_left': 36,
                 'left_eye_right': 39,
                 'right_eye_left': 42,
                 'right_eye_right': 45,
                 'nose_mid': 29,
                 'nose_tip': 33}


def get_roi_coordinates(landmarks) -> dict:
    """ Gets necessary coordinates of points which will serve as boundaries. """

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
    left_roi = image[roi_coords['left_1']:roi_coords['left_2'], roi_coords['top']:roi_coords['bottom']]  # todo this does not work properly yet
    right_roi = image[roi_coords['right_1']:roi_coords['right_2'], roi_coords['top']:roi_coords['bottom']]

    return left_roi, right_roi


if __name__ == "__main__":
    directories.change_dir_to_main()

    model_dir = directories.face_68_model_dir
    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor(model_dir)

    # We now reading image on which we applied our face detector
    image = "data/examples/exampleface.jpg"
    img = cv2.imread(image)
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    left_roi, right_roi = get_roi_images(face_detector, landmark_detector, imageRGB)

    cv2.imshow("Face landmark result", left_roi)
    cv2.imshow("Face landmark result", right_roi)

    # Pause screen to wait key from user to see result
    cv2.waitKey(0)
    cv2.destroyAllWindows()