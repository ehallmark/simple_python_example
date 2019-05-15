import argparse
import json
import os
from pathlib import Path
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import sys

output_dir = sys.argv[1]
features_path = os.path.join(output_dir, 'features.txt')
labels_path = os.path.join(output_dir, 'labels.txt')
model_path = os.path.join(output_dir, 'model.keras')

x = np.loadtxt(features_path)
y = np.loadtxt(labels_path)
y = to_categorical(y, 2)

model = load_model(model_path)

accuracy = model.evaluate(x, y, verbose=0)[1]

metrics = {
    'metrics': [{
      'name': 'accuracy-score', # The name of the metric. Visualized as the column name in the runs table.
      'numberValue':  accuracy, # The value of the metric. Must be a numeric value.
      'format': "PERCENTAGE",   # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
    }]
}
with open('/mlpipeline-metrics.json', 'w') as f:
    json.dump(metrics, f)

print("Accuracy for model: ", accuracy)
