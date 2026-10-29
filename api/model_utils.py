"""Inference + Grad-CAM for the team's Keras DenseNet121 multi-label model.

Matches the notebook exactly:
  RGB -> tf.image.resize to img_size -> / 255.   (ImageNet normalization is
  INSIDE the model), sigmoid outputs, per-class thresholds from
  inference_config.json.
"""
import base64
import io
import json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import tensorflow as tf
from PIL import Image

NORMAL_CLASS = "No Finding"


@dataclass
class Bundle:
    model: tf.keras.Model
    classes: List[str]
    thresholds: np.ndarray
    img_size: int
    model_file: str
    # pieces used for Grad-CAM
    norm: tf.keras.layers.Layer
    base: tf.keras.Model
    head: List[tf.keras.layers.Layer]


def load_bundle(model_path: str, config_path: str) -> Bundle:
    cfg = json.load(open(config_path))
    model = tf.keras.models.load_model(model_path, compile=False)
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    idx = model.layers.index(base)
    return Bundle(
        model=model,
        classes=cfg["classes"],
        thresholds=np.asarray(cfg["thresholds"], dtype="float32"),
        img_size=cfg["img_size"],
        model_file=cfg.get("model_file", model_path.split("/")[-1]),
        norm=model.get_layer("imagenet_norm"),
        base=base,
        head=model.layers[idx + 1:],   # gap, dropout, dense
    )


def preprocess(bundle: Bundle, img: Image.Image) -> tf.Tensor:
    arr = tf.convert_to_tensor(np.asarray(img.convert("RGB")), dtype=tf.float32)
    arr = tf.image.resize(arr, [bundle.img_size, bundle.img_size]) / 255.0
    return arr[None, ...]


def predict_probs(bundle: Bundle, x: tf.Tensor) -> np.ndarray:
    return bundle.model(x, training=False).numpy()[0]


def review_priority(findings, high_ratio: float) -> str:
    """Rule-based placeholder: justify/tune high_ratio from validation results.

    High   = an abnormal class is detected and its prob/threshold >= high_ratio
    Medium = abnormal detected below that ratio, or nothing clearly detected
    Low    = only 'No Finding' is detected
    """
    abnormal = [f for f in findings if f.detected and f.disease != NORMAL_CLASS]
    if abnormal:
        top = max(f.probability / max(f.threshold, 1e-6) for f in abnormal)
        return "High" if top >= high_ratio else "Medium"
    normal = next((f for f in findings if f.disease == NORMAL_CLASS), None)
    return "Low" if normal is not None and normal.detected else "Medium"


def default_explain_class(bundle: Bundle, probs: np.ndarray) -> str:
    ratios = probs / np.maximum(bundle.thresholds, 1e-6)
    order = np.argsort(-ratios)
    for i in order:
        if bundle.classes[i] != NORMAL_CLASS:
            return bundle.classes[i]
    return bundle.classes[int(order[0])]


def gradcam_base64(bundle: Bundle, x: tf.Tensor, class_name: str,
                   original: Image.Image) -> str:
    """Grad-CAM on DenseNet121's final feature map for one class (sigmoid score)."""
    k = bundle.classes.index(class_name)
    with tf.GradientTape() as tape:
        conv = bundle.base(bundle.norm(x), training=False)
        tape.watch(conv)
        h = conv
        for layer in bundle.head:
            h = layer(h, training=False)
        score = h[:, k]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1))[0].numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    base_img = original.convert("RGB")
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(base_img.size, Image.BILINEAR)
    c = np.asarray(cam_img, dtype=np.float32) / 255.0
    heat = np.zeros((*c.shape, 3), dtype=np.float32)
    heat[..., 0] = np.clip(c * 2, 0, 1)
    heat[..., 1] = np.clip(c * 2 - 1, 0, 1)
    base = np.asarray(base_img, dtype=np.float32) / 255.0
    overlay = np.clip(0.6 * base + 0.4 * heat, 0, 1)
    buf = io.BytesIO()
    Image.fromarray((overlay * 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
