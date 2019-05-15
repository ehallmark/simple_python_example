import argparse
import json
import os
from pathlib import Path
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import sys

output_dir = sys.argv[1]
num_examples = 10000

features_path = os.path.join(output_dir, 'features.txt')
labels_path = os.path.join(output_dir, 'labels.txt')
output_model_path = os.path.join(output_dir, 'model.keras')

x_train = np.loadtxt(features_path)
y_train = np.loadtxt(labels_path)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

metrics = []

y_train = keras.utils.to_categorical(y_train, 2)
metrics.append('accuracy')

learning_rate = 0.001
decay = 1e-6
loss = 'categorical_crossentropy'

opt = keras.optimizers.Adam(lr=learning_rate, decay=decay)

model = Sequential([
    Dense(50, input_shape=(2,), activation='tanh'),
    Dense(2, activation='softmax')
])

# Let's train the model using RMSprop
model.compile(loss=loss,
              optimizer=opt,
              metrics=metrics)

x_train = x_train.astype('float32')

model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=4,
    shuffle=True
)

# Save model and weights
if not output_model_path.startswith('gs://'):
    save_dir = os.path.dirname(output_model_path)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

model.save(output_model_path)
print('Saved trained model at %s ' % output_model_path)
