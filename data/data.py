import json

from project import BASE_DIR

import os
import numpy as np
import glob


def get_output_keys():
    directories = glob.glob(os.path.join(BASE_DIR, "input", "model_outputs", "*"))
    directories = [d for d in directories if os.path.isdir(d)]
    model_keys = [os.path.basename(d) for d in directories]
    return model_keys


def load_output(output_key):
    model_dir = os.path.join(BASE_DIR, "input", "model_outputs", output_key)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError("Model outputs for {} do not exist in {}".format(output_key, model_dir))

    npy_paths = {
        "gf": os.path.join(model_dir, "gf.npy"),
        "qf": os.path.join(model_dir, "qf.npy"),
    }
    json_keys = [
        "g_pids", "q_pids", "g_camids", "q_camids",
    ]
    json_path = os.path.join(model_dir, "metadata.json")

    outputs = {}
    for key, path in npy_paths.items():
        outputs[key] = np.load(path)
    with open(json_path, "r") as f:
        d = json.load(f)
        if set(d.keys()) != set(json_keys):
            raise Exception("Corrupt metadata file {}".format(json_path))
        outputs.update(d)

    return outputs
