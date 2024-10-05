import numpy as np 
import json
import os 
import math 
import torch
from torch import nn
import torch.nn.functional as F

#################################################

def add_predictions_n_probabilities_df(predictions, df):
    df['prediction'] = list(predictions.predictions.argmax(-1))
    y_pred_pt = torch.from_numpy(predictions.predictions)
    probs = nn.functional.softmax(y_pred_pt, dim=-1)
    max_probabilities, max_indices = torch.max(probs, dim=1)
    df['probability'] = list(max_probabilities)
    return df

def calculate_warmup_steps(train_size, num_epochs:int = 8, batch_size:int = 8, step_fraction:float = 0.1):
    total_steps = (train_size // batch_size) * num_epochs
    return math.floor(step_fraction* total_steps)

#################################################

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def dump_configuration_to_json(training_config):
    json_config = json.dumps(training_config, indent='\t\t', cls=NpEncoder)
    json_file_name = os.path.join(training_config['training_output_path'], 'training_configuration.json')
    with open(json_file_name, 'w', encoding='utf8') as json_file:
        json_file.write(json_config)
    print('saved training configuration at: ' + json_file_name)
    

