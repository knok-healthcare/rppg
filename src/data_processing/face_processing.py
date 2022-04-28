import dlib, cv2
import numpy as np
import directories

# Landmark's numbers according to dlib's 68 model
roi_landmarks = {'left_eye_left': 37,
                 'left_eye_right': 40,
                 'right_eye_left': 43,
                 'right_eye_right': 46,
                 'nose_mid': 30,
                 'nose_tip': 34}


def get_roi_coordinates(landmarks: list) -> dict:
    """ Gets necessary coordinates of points which will serve as boundaries. """
    return {'top': landmarks[roi_landmarks['nose_mid']][1],
            'bottom': landmarks[roi_landmarks['nose_tip']][1],
            'left_1': landmarks[roi_landmarks['left_eye_left']][0],
            'left_2': landmarks[roi_landmarks['left_eye_right']][0],
            'right_1': landmarks[roi_landmarks['right_eye_left']][0],
            'right_2': landmarks[roi_landmarks['right_eye_right']][0]}


def draw_points(image, face_landmarks):
    """ Draws landmarks on the face. Useful for debug and presentation. """
    for i in range(face_landmarks):
        point = [face_landmarks.part(i).x, face_landmarks.part(i).y]
        cv2.circle(image, point, radius=4, color=(0, 0, 255), thickness=-1)


if __name__ == "__main__":
    directories.change_dir_to_main()

    model_dir = directories.face_68_model_dir
    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor(model_dir)

    # We now reading image on which we applied our face detector
    image = "data/examples/exampleface.jpg"

    # Now we are reading image using openCV
    img = cv2.imread(image)
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Now this line will try to detect all faces in an image either 1 or 2 or more faces
    face = face_detector(imageRGB, 0)[0]
    face_rectangle = dlib.rectangle(int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
    detected_landmarks = landmark_detector(imageRGB, face_rectangle)

    print("Total number of face landmarks detected ", len(detected_landmarks.parts()))

    cv2.imshow("Face landmark result", img)

    # Pause screen to wait key from user to see result
    cv2.waitKey(0)
    cv2.destroyAllWindows()
