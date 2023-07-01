import pickle
import os
def open_object(object_path):
    with open(object_path, mode='rb') as f:
        obj = pickle.load(f)

    return obj


def save_object(object_path, obj):
    os.makedirs(os.path.dirname(object_path),exist_ok=True)
    with open(object_path, mode='wb') as f:
        pickle.dump(obj, f)