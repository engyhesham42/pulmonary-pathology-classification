"""Create an UNTRAINED placeholder with the SAME architecture/config format as
the team's model, so the API/UI/Docker pipeline can be tested end to end.
Delete these two files and use the team's real ones afterwards.
"""
import json
import os

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121

CLASSES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
           'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 'No Finding', 'Nodule',
           'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
IMG = 224

inputs = layers.Input((IMG, IMG, 3), name="xray_image")
x = layers.Normalization(mean=[0.485, 0.456, 0.406],
                         variance=[0.229**2, 0.224**2, 0.225**2], name="imagenet_norm")(inputs)
base = DenseNet121(weights=None, include_top=False, input_shape=(IMG, IMG, 3))
x = base(x, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.3, name="dropout")(x)
out = layers.Dense(len(CLASSES), activation="sigmoid", name="disease_probabilities")(x)
model = Model(inputs, out, name="xray_densenet121")

os.makedirs("models", exist_ok=True)
model.save("models/run3_finetuned.keras")
json.dump({"model_file": "run3_finetuned.keras", "img_size": IMG,
           "input": "RGB, resized, /255", "classes": CLASSES,
           "thresholds": [0.5] * len(CLASSES)},
          open("models/inference_config.json", "w"), indent=2)
print("wrote UNTRAINED placeholder model + config in models/")
