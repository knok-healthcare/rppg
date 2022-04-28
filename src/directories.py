from pathlib import Path
import os

project_dir = Path(os.path.abspath(__file__)).parent.parent
face_68_model_dir = "models/shape_predictor_68_face_landmarks.dat"


def change_dir_to_main(destination=project_dir):
    # Method necessary to be used in scripts before importing data_processing and scripts from different folders.
    if os.getcwd() != project_dir:
        os.chdir(destination)