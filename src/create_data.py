import argparse
import json
import os
from pathlib import Path
import numpy as np
import sys

output_dir = sys.argv[1]
num_examples = 10000

features_path = os.path.join(output_dir, 'features.txt')
labels_path = os.path.join(output_dir, 'labels.txt')

features = np.random.uniform(low=0.0, high=1.0, size=(num_examples, 2))
features = (features > 0.5)
labels = np.logical_xor(features[:, 0], features[:, 1]).astype(np.int32)
features = features.astype(np.float32)

save_dir = os.path.dirname(features_path)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

np.savetxt(labels_path, labels)
np.savetxt(features_path, features)
print("Saved labels and features.")
