from fastai.vision.all import Path
from fastbook import load_learner

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = temp # !WARNING: change to pathlib.WindowsPath when used on Windows

def load_model():
    custom_path = Path("./models/cv_initial_model.pkl")
    learn = load_learner(custom_path)

    return learn
