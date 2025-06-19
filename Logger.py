from argparse import Namespace
from datetime import datetime
from random import choices
from typing_utils import *

import json
import numpy as np
import os
import pandas as pd
import string
import torch
import yaml

class LogImage:
    def __init__(self, data):
        self.data = data
    
    def to_uint8(self):
        dmin = np.min(self.data)

        if dmin < 0:
            self.data = (self.data - dmin) / np.ptp(self.data)
        if np.max(self.data) <= 1:
            self.data = (self.data * 255).astype(np.int32)
        
        return self.data.clip(0, 255).astype(np.uint8)

class Logger:
    def __init__(self, name: str = ''):
        base_log_dir = os.path.join(os.path.curdir, 'logs')

        exp_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_hash = ''.join(choices(string.ascii_letters + string.digits, k=4))
        exp_dir = os.path.join(base_log_dir, f"run-{exp_date}-{name}-{exp_hash}")

        try:
            if not os.path.exists(base_log_dir):
                os.mkdir(base_log_dir)
        except:
            raise SystemError(f"Could not make new directory: {base_log_dir}")

        try:
            while os.path.exists(exp_dir):
                exp_hash = ''.join(choices(string.ascii_letters+string.digits, k=8))
                exp_dir = os.path.join(base_log_dir, f"run-{exp_date}-{name}-{exp_hash}")    
            os.mkdir(exp_dir)
            exp_media = os.path.join(exp_dir, 'media')
        except:
            raise SystemError(f"Could not make new directory: {exp_dir}")

        try:
            if not os.path.exists(exp_media):
                os.mkdir(exp_media)
            else:
                raise RuntimeError()
        except:
            raise SystemError(f"Could not make new directory: {exp_media}")

        try:
            exp_tables = os.path.join(exp_media, 'table')
            
            if not os.path.exists(exp_tables):
                os.mkdir(exp_tables)
            else:
                raise RuntimeError()
        except:
            raise SystemError(f"Could not make new sub directories: \n{exp_tables}")


        self.name = name
        
        self.base_dir = exp_dir
        self.summary_file = os.path.join(exp_dir, 'summary.json')
        self.config_file = os.path.join(exp_dir, 'config.yaml')
        self.media_dir = exp_media
        self.table_dir = exp_tables

        self.logs = []

    def create_table(self, columns: dict):
        self.columns = columns
        self.data = []
        self.image_dirs = []
        
        for key in self.columns:
            if self.columns[key] == LogImage:
                path = os.path.join(self.media_dir, key)
                os.mkdir(path)
                self.columns[key] = path
            if self.columns[key] == torch.Tensor:
                path = os.path.join(self.media_dir, key)
                os.mkdir(path)
                self.columns[key] = path
        
        self.table_dir = os.path.join(self.table_dir, 'metadata.csv')

    def config(self, args: Namespace):
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(args.__dict__, f)

    def add_data(self, data):
        if len(data) != len(self.columns):
            raise ValueError(
                "Table expects {} columns: {}, found {}".format(
                    len(self.columns), self.columns, len(data)
                )
            )
        self.data.append(data)

    def get_data(self, key):
        i = list(self.columns.keys()).index(key)
        data = []
        for row in self.data:
            data.append(row[i])
        return data

    def log(self, data: dict):
        self.logs.append(data)
    
    def close(self):
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f)
        
        tensor_data = dict()
        for i, row in enumerate(self.data):
            tmp_row = list(row)
            for j, item in enumerate(tmp_row):
                if isinstance(item, LogImage):
                    path = self.columns[list(self.columns.keys())[j]]
                    img = item.data
                    img.save(os.path.join(path, f"{i}.png"),'PNG')
                    tmp_row[j] = i
                if isinstance(item, torch.Tensor):
                    key = list(self.columns.keys())[j]
                    if tensor_data.get(key):
                        tensor_data[key].append(item)
                    else:
                        tensor_data[key] = [item]
                    tmp_row[j] = i
            self.data[i] = tmp_row
        
        if tensor_data:
            for key in tensor_data:
                tensors = torch.stack(tensor_data[key])
                idxs = list(range(len(self.data)))
                torch.save((idxs,tensors), os.path.join(self.columns[key],"data.pt"))

        df = pd.DataFrame(self.data, columns=self.columns)
        df.to_csv(self.table_dir)