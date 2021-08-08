import os
import json

from tqdm.auto import tqdm
import pandas as pd

def prepare_dir(dir_name):
    if not os.path.exists(dir_name): os.makedirs(dir_name)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data

def dump_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        print("Data file is dumped at ", file_path)


def load_json_lines(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_json_lines_with_progress(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            data.append(json.loads(line))
    return data


def load_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def dump_csv(file_path, df):
    df.to_csv(file_path, index=False)
    print("Data file is dumped at ", file_path)


def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        data = []
        for line in lines:
            data.append(line.strip())
        return data

def dump_txt(file_path, lines):
    with open(file_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
        print("Data file is dumped at ", file_path)