# Multi-Label Chest X-Ray Disease Classification

Samsung Innovation Campus (SIC) - Artificial Intelligence Track - Graduation Project

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Repository Structure](#3-repository-structure)
4. [Dataset](#4-dataset)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Data Preparation and Augmentation](#6-data-preparation-and-augmentation)
7. [Model and Training](#7-model-and-training)
8. [Experiment Tracking](#8-experiment-tracking)
9. [Results](#9-results)
10. [Explainability (Grad-CAM)](#10-explainability-grad-cam)
11. [Application and Deployment](#11-application-and-deployment)
12. [How to Reproduce](#12-how-to-reproduce)
13. [Limitations](#13-limitations)
14. [Team](#14-team)
15. [References](#15-references)

---

## 1. Introduction

The chest X-ray is the most frequently performed medical imaging examination and is usually the first
step in diagnosing many lung and heart conditions. Interpreting it is time-consuming and depends on
the experience of the radiologist. A single image may show several findings at once, and early or subtle
findings, such as small nodules or mild infiltration, are easy to overlook under a high workload.

This project develops a deep learning system that analyses a frontal chest X-ray and indicates which of
13 thoracic diseases may be present, in addition to a "No Finding" class. The system is built on a
DenseNet121 convolutional neural network pretrained on ImageNet and fine-tuned on the NIH ChestX-ray14
dataset. All experiments are tracked with MLflow, model decisions are visualised with Grad-CAM, and the
results are presented through a web application.

The system is intended as a decision-support tool that helps clinicians prioritise cases. It is not
intended to replace a qualified radiologist.

## 2. Problem Statement

Given a frontal chest X-ray, the task is to predict every finding present in the image. Because one
image can carry several labels at the same time, this is a **multi-label classification** problem with
14 outputs.

The task presents three main challenges:

- **Severe class imbalance.** Several diseases (for example Pneumonia, Edema and Fibrosis) appear in
  only 1 to 5 percent of the images, while "No Finding" dominates the validation and test sets.
- **Noisy labels.** The labels of ChestX-ray14 were extracted automatically from radiology reports
  using natural language processing, so a portion of them is inaccurate.
- **Asymmetric cost of errors.** In a screening context, a missed disease (false negative) is
  considerably more harmful than a false alarm (false positive).

The objective of the project is therefore a model with a **high recall**, which detects as many true
cases as possible, while keeping precision at an acceptable level. Ranking quality is measured with
ROC-AUC, and a separate decision threshold is selected for each disease to favour recall.

## 3. Repository Structure

```
.
|-- README.md
|-- EDA.ipynb                        Exploratory data analysis (before and after augmentation)
|-- grad_project_improved (1).ipynb  Training of Runs 1-3 (baseline, frozen backbone, full fine-tuning)
|-- final_mode.ipynb                 Final model (Run 4): recall-oriented training and evaluation
|-- XAI_GradCAM_Run3 (1).ipynb       Grad-CAM explainability and confusion matrices
|-- final_csv.csv                    Final training metadata (26,530 images, equivalent to combined.csv)
|-- final_model-*.zip                Final model and all evaluation outputs (see Section 9.4)
|-- Architecture.docx                Architecture of the web application
|-- data_links.docx                  Links to the image archives used in this project
|-- link for the models.docx         Link to the trained model files
|-- link for the docker file.docx    Link to the Docker image / Dockerfile of the application
|-- Video 2026-10-29 at 4.08.30 AM.mp4   Demonstration video of the application
```

Large files (image archives, trained models and the Docker image) are not stored in the repository
because of their size. They are available through the links in the corresponding documents.

## 4. Dataset

The project uses a subset of the **NIH ChestX-ray14** dataset released by the National Institutes of
Health Clinical Center. Each record contains the image file name, the finding labels separated by
`|`, the patient identifier, age, gender, view position (PA or AP) and the original image size.

| Split | Images | Source file | Notes |
|---|---|---|---|
| Training | 26,530 | `combined.csv` (`final_csv.csv`) | 14,950 original images, 7,204 additional NIH images, 4,376 offline-augmented images |
| Validation | 5,646 | `validation/validation.csv` | Original images only |
| Test | 5,369 | `test/test.csv` | Original images only |

All images were resized to 512 x 512 pixels for storage and are resized to 224 x 224 pixels at
training time.

**Classes (14 outputs):** Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema,
Fibrosis, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax and No Finding.

The **Hernia** class was excluded because it contained too few images (30 records in the training
set) to be learned reliably. The Hernia label is removed from the affected records while the remaining
labels of those images are kept.

## 5. Exploratory Data Analysis

The analysis is documented in `EDA.ipynb`. The main findings on the original training set are:

- **Size:** 14,950 images from 4,529 unique patients, with no missing values and no duplicated
  records or file names.
- **Images per patient:** a mean of 3.3 and a median of 1. A total of 2,169 patients have more than one
  image (up to 73), which makes a patient-level split necessary to prevent data leakage.
- **Labels:** 5,000 images (33.4 percent) are labelled "No Finding" and 9,950 images (66.6 percent)
  carry at least one disease label. No image is labelled both "No Finding" and a disease.
- **Demographics:** a median patient age of 50 years; 8,208 images of male patients and 6,742 of
  female patients. One age value (412) is a recording error.
- **View position:** 8,771 PA images and 6,179 AP images. Disease prevalence differs between the two
  views, which reflects different patient populations rather than a causal effect.
- **Co-occurrence:** several diseases frequently appear together (for example Effusion with
  Infiltration and Atelectasis), which confirms the multi-label nature of the task.

## 6. Data Preparation and Augmentation

### 6.1 Offline augmentation

To reduce class imbalance, 4,376 augmented images were generated offline for the nine minority classes:
Mass (584), Pleural Thickening (520), Pneumothorax (515), Emphysema (500), Cardiomegaly (477),
Edema (477), Consolidation (446), Fibrosis (432) and Pneumonia (425). Augmented images are identified
by the suffix `_aug_<Class>_<n>` in their file name. Additional original NIH images were also added
to the training set.

The effect on class balance, measured in `EDA.ipynb`, is summarised below.

| Metric (all labels) | Before | After | Change |
|---|---|---|---|
| Max / min class ratio | 18.18 | 6.21 | -65.8% |
| Coefficient of variation | 0.940 | 0.637 | -32.2% |
| Gini coefficient | 0.464 | 0.323 | -30.4% |

The classes that benefited most were Pneumonia (+193 percent), Edema (+187 percent), Emphysema
(+127 percent), Fibrosis (+109 percent) and Pleural Thickening (+100 percent). The share of
"No Finding" decreased from 25.3 to 17.8 percent of all label instances.

### 6.2 Data cleaning and leakage control

- Records are deduplicated by file name, and every file name is matched to an image on disk.
- The patient identifier is extracted from each file name, including augmented ones.
- Any training image belonging to a validation or test patient is removed. The check confirmed
  that no patient is shared between the training, validation and test sets.

### 6.3 Input pipeline

The input pipeline is built with `tf.data`. Images are decoded, resized to 224 x 224, scaled to the
range [0, 1] and fully shuffled at every epoch. ImageNet mean and standard deviation normalisation is
performed inside the model, so the deployed model accepts raw pixel values divided by 255.

In the final model, online augmentation (random brightness, random contrast and a random crop of
90 to 100 percent of the image, which acts as a zoom and shift) is applied **only to original
images**. The offline-augmented images are not augmented a second time, so that subtle findings are
not distorted beyond recognition.

## 7. Model and Training

### 7.1 Architecture

```
Input (224 x 224 x 3)
  -> ImageNet normalisation
  -> DenseNet121 backbone (pretrained on ImageNet)
  -> Global Average Pooling
  -> Dropout (0.3)
  -> Dense (14 units, sigmoid)
```

DenseNet121 was selected because it is the backbone used by CheXNet, a reference model on this
dataset, and it offers a good balance between accuracy and size (about 7 million parameters).

### 7.2 Training runs

The model was developed through four tracked runs. Each run addressed a weakness of the previous one.

| Run | Description | Loss | Learning rate | Best validation ROC-AUC |
|---|---|---|---|---|
| Run 1 | Baseline, inputs scaled to [-1, 1] | Binary cross-entropy | 1e-3 | 0.685 (last epoch) |
| Run 2 | Frozen backbone, corrected ImageNet normalisation | Weighted binary cross-entropy | 1e-3 | 0.705 |
| Run 3 | Full fine-tuning from Run 2, BatchNorm layers frozen | Weighted binary cross-entropy | 1e-4 | 0.780 |
| Run 4 (final) | Fine-tuning from Run 3 with recall-oriented changes | Weighted focal loss | 5e-5 | 0.771 |

**Class weighting.** Each class receives a positive weight equal to the ratio of negative to positive
samples, limited to the range [1, 20].

**Weighted focal loss (Run 4).** The focal term (gamma = 2) reduces the contribution of easy examples,
so that training concentrates on positive cases the model still misses. The class weights are
applied on top of the focal term.

**Callbacks.** Model checkpointing and early stopping (patience 3) monitor the validation macro
ROC-AUC, and the learning rate is reduced by a factor of 0.3 when it stops improving.

### 7.3 Threshold selection

Because the model outputs one probability per disease, a decision threshold is required for each
class. Thresholds are tuned on the validation set and then applied, unchanged, to the test set.
Four strategies were compared:

- **0.5:** a fixed threshold for every class.
- **F1-max:** the threshold that maximises F1. This was the method used in Run 3. It produced high
  thresholds (0.54 to 0.89) and therefore a low recall.
- **F2-max:** the threshold that maximises F2, which weights recall twice as much as precision.
  **This strategy was selected for the final model.**
- **Recall >= 0.75:** the highest threshold that still achieves a recall of at least 0.75 on the
  validation set.

## 8. Experiment Tracking

All runs are logged with **MLflow** in the experiment `Chest_XRay_MultiLabel`, stored in a SQLite
database (`mlflow.db`). For each run the following are recorded: hyperparameters, per-epoch training
and validation metrics, final test metrics and output artifacts. The runs can be compared with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 9. Results

All results below are measured on the held-out **test set** (5,369 images).

### 9.1 Effect of the threshold strategy (final model)

| Threshold strategy | ROC-AUC | PR-AUC | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|---|
| 0.5 | 0.765 | 0.227 | 0.533 | 0.181 | 0.249 | 0.822 |
| F1-max (previous method) | 0.765 | 0.227 | 0.382 | 0.230 | 0.277 | 0.886 |
| **F2-max (selected)** | **0.765** | **0.227** | **0.603** | **0.168** | **0.248** | **0.768** |
| Recall >= 0.75 | 0.765 | 0.227 | 0.726 | 0.156 | 0.222 | 0.676 |

All values are macro averages over the 14 classes. Accuracy is the mean per-class accuracy.

The selected strategy raises the macro recall from **0.38 to 0.60** compared with the previous
method, at the cost of a lower precision and accuracy. This trade-off was made deliberately, since a
missed disease is the most costly error in screening. The "Recall >= 0.75" strategy is available when
an even higher sensitivity is required.

Accuracy is reported for completeness but is not used as the main metric. Because most images are
negative for any given disease, a model that never predicts a disease would also obtain a high
accuracy. ROC-AUC and recall are more informative for this problem.

### 9.2 Per-class results (final model, F2-max thresholds)

| Disease | ROC-AUC | PR-AUC | Recall | Precision | F1 | Threshold | Positives |
|---|---|---|---|---|---|---|---|
| Atelectasis | 0.777 | 0.281 | 0.745 | 0.189 | 0.301 | 0.481 | 506 |
| Cardiomegaly | 0.930 | 0.314 | 0.708 | 0.219 | 0.334 | 0.498 | 130 |
| Consolidation | 0.757 | 0.099 | 0.714 | 0.080 | 0.144 | 0.463 | 189 |
| Edema | 0.814 | 0.072 | 0.351 | 0.063 | 0.107 | 0.470 | 74 |
| Effusion | 0.859 | 0.409 | 0.761 | 0.291 | 0.421 | 0.475 | 506 |
| Emphysema | 0.850 | 0.178 | 0.495 | 0.184 | 0.268 | 0.604 | 95 |
| Fibrosis | 0.720 | 0.074 | 0.489 | 0.051 | 0.092 | 0.450 | 131 |
| Infiltration | 0.713 | 0.276 | 0.820 | 0.201 | 0.322 | 0.418 | 755 |
| Mass | 0.736 | 0.125 | 0.281 | 0.142 | 0.188 | 0.534 | 196 |
| No Finding | 0.747 | 0.801 | 0.997 | 0.603 | 0.751 | 0.269 | 3,224 |
| Nodule | 0.667 | 0.144 | 0.610 | 0.078 | 0.139 | 0.502 | 282 |
| Pleural Thickening | 0.702 | 0.089 | 0.379 | 0.096 | 0.153 | 0.532 | 153 |
| Pneumonia | 0.613 | 0.048 | 0.519 | 0.014 | 0.027 | 0.374 | 54 |
| Pneumothorax | 0.830 | 0.259 | 0.575 | 0.138 | 0.223 | 0.512 | 181 |

The strongest classes are Cardiomegaly (ROC-AUC 0.93), Effusion (0.86), Emphysema (0.85) and
Pneumothorax (0.83), which have clear visual signs. The weakest are Pneumonia (0.61) and Nodule
(0.67), which are subtle and rare in the test set.

### 9.3 Recall per class before and after the recall-oriented changes

| Disease | Run 3 recall (F1-max) | Final recall (F2-max) |
|---|---|---|
| Atelectasis | 0.504 | 0.745 |
| Cardiomegaly | 0.569 | 0.708 |
| Consolidation | 0.228 | 0.714 |
| Edema | 0.189 | 0.351 |
| Effusion | 0.518 | 0.761 |
| Emphysema | 0.432 | 0.495 |
| Fibrosis | 0.122 | 0.489 |
| Infiltration | 0.494 | 0.820 |
| Mass | 0.189 | 0.281 |
| Nodule | 0.213 | 0.610 |
| Pleural Thickening | 0.294 | 0.379 |
| Pneumonia | 0.185 | 0.519 |
| Pneumothorax | 0.481 | 0.575 |

Recall improved for every disease, with the largest gains for Consolidation, Fibrosis, Nodule and
Pneumonia.

### 9.4 Output files

The final model and its evaluation outputs are provided in `final_model-*.zip`:

| File | Content |
|---|---|
| `recall_boost_densenet121.keras` | Trained final model |
| `inference_config_recall.json` | Class names, input specification and the selected per-class thresholds |
| `test_summary_by_threshold.csv` | Macro test metrics for each threshold strategy |
| `test_per_class_metrics.csv` | Per-class test metrics for each threshold strategy |
| `test_probabilities.npy` | Predicted test probabilities, used to re-tune thresholds without re-running the model |
| `training_history.csv`, `training_history.png` | Loss, ROC-AUC and recall per epoch |
| `recall_per_class_old_vs_new.png` | Per-class recall before and after the changes |
| `roc_pr_curves.png` | ROC and precision-recall curves for every class |
| `mlflow.db` | MLflow tracking database containing all runs |

## 10. Explainability (Grad-CAM)

`XAI_GradCAM_Run3 (1).ipynb` applies **Grad-CAM** to the fine-tuned model to show which regions of
the X-ray drive each prediction. The heatmap is computed from the last DenseNet121 feature map
(7 x 7 x 1024) using the gradient of the selected disease's logit, so that confident predictions
still produce informative maps.

The notebook produces:

- One correctly detected example per disease, to show what the model relies on when it is correct.
- Random test images with the heatmaps of the three highest-scoring diseases.
- Missed cases (1,935 image-disease pairs for Run 3), to analyse the limits of the model.
- A confusion matrix for every disease, with true positives, false negatives, false positives,
  true negatives, recall, precision and specificity.
- A function that explains any uploaded X-ray.

Grad-CAM indicates the regions that influenced the model. It does not prove the anatomical location
of a disease.

## 11. Application and Deployment

The architecture of the application is described in `Architecture.docx`.

```
Streamlit (UI) -> FastAPI (backend) -> Model inference + Grad-CAM
                                              |
                     Fixed specialty table + Groq LLM -> Explanation
```

- **User interface (Streamlit):** accepts PNG and JPG uploads and displays the original X-ray next to
  its Grad-CAM heatmap, with a selector for the condition to visualise. It shows the conditions above
  the decision threshold and score bars for all classes. The interface communicates only with the API.
- **Backend (FastAPI):** runs model inference and Grad-CAM. The language model logic is kept in a
  separate module (`backend/llm_service.py`) and is not mixed with the inference code.
- **Explanation (Groq LLM):** generates four short paragraphs (what the condition is, what the score
  means, what the heatmap shows, and the limits of the result). The language model receives only class
  names, scores, the threshold, the selected class and approved specialty names. No image and no
  patient data are sent. It never changes a score, a label or a referral.
- **Safety controls:** a strict system prompt, a temperature of 0 with a fixed seed, and output
  validation with a Pydantic schema and rule checks. These reject any mention of medication or
  treatment, urgency or severity, unsupplied specialties, numbers not present in the input, or
  claims of examining the image. On failure, the service retries once and then displays
  "explanation unavailable" without affecting the predictions or the heatmap.
- **Suggested follow-up:** general specialties (primary care, pulmonology, cardiology) are taken from a
  fixed table (`backend/guidance.py`), not from the language model.
- **Disclaimer:** a research-use disclaimer is displayed at the top and bottom of the page.

### 11.1 Configuration

The application is configured through environment variables. A `.env.example` file without secrets
is included.

| Variable | Purpose |
|---|---|
| `API_URL` | Address of the FastAPI backend used by the interface |
| `GROQ_API_KEY` | API key for the Groq service |
| `GROQ_MODEL` | Language model name (default: `openai/gpt-oss-120b`) |
| `GROQ_TIMEOUT` | Request timeout for the language model |
| `DECISION_THRESHOLD` | Decision threshold applied to model scores |

### 11.2 Running with Docker

The Docker image is available at the link in `link for the docker file.docx`.

```bash
docker pull <docker-image-name>
docker run -p 8501:8501 --env-file .env <docker-image-name>
```

The application is then available at `http://localhost:8501`. A demonstration of the application is
provided in the video file included in the repository.

## 12. How to Reproduce

### 12.1 Requirements

- Google Colab with a GPU runtime
- TensorFlow 2.x, scikit-learn, pandas, NumPy, Matplotlib and MLflow
- A Google Drive account with enough free space for the image archives and models

### 12.2 Data setup

Download the files listed in `data_links.docx` and `link for the models.docx`, and place them in
Google Drive with the following structure:

```
MyDrive/
`-- x_ray_ds/
    |-- archive (1).zip          Training, validation and test images (512 x 512) with their CSV files
    |-- images_5_6.zip           Additional NIH images
    |-- combined.csv             Final training metadata (same content as final_csv.csv)
    `-- models/
        `-- run3_finetuned.keras Optional; the final notebook starts from this checkpoint if present
```

The notebooks extract the archives to the Colab local disk automatically the first time they are run.

### 12.3 Execution order

| Step | Notebook | Purpose | Approximate time on a Colab GPU |
|---|---|---|---|
| 1 | `EDA.ipynb` | Explore the data and measure the effect of augmentation | A few minutes |
| 2 | `grad_project_improved (1).ipynb` | Train Runs 1 to 3 and log them to MLflow | 1 to 2 hours |
| 3 | `final_mode.ipynb` | Train and evaluate the final model, export all outputs | 30 to 60 minutes when starting from Run 3 |
| 4 | `XAI_GradCAM_Run3 (1).ipynb` | Generate Grad-CAM visualisations and confusion matrices | About 15 minutes |

For each notebook, select **Runtime > Change runtime type > GPU**, then **Runtime > Run all**. The
final notebook requests Google Drive authorisation at the start in order to upload its outputs.

To change the operating point of the final model without retraining, set `THRESHOLD_MODE` to
`"f2"` or `"recall_target"` (and adjust `TARGET_RECALL`), then re-run Sections 8 to 10 of
`final_mode.ipynb` only.

### 12.4 Inference

```python
import json
import tensorflow as tf

model  = tf.keras.models.load_model("recall_boost_densenet121.keras", compile=False)
config = json.load(open("inference_config_recall.json"))

img = tf.image.decode_image(tf.io.read_file("xray.png"), channels=3, expand_animations=False)
img = tf.image.resize(img, [config["img_size"], config["img_size"]]) / 255.0
prob = model.predict(img[None, ...], verbose=0)[0]

for name, p, t in zip(config["classes"], prob, config["thresholds"]):
    print(f"{name:<20} {p:.3f}  {'DETECTED' if p >= t else ''}")
```

## 13. Limitations

- The labels of ChestX-ray14 were mined from reports and contain noise, which limits the maximum
  achievable performance.
- The class distribution differs strongly between splits: "No Finding" represents 18.8 percent of the
  training images but 55.8 percent of the validation and 60.0 percent of the test images.
- Precision remains low for rare diseases such as Pneumonia and Fibrosis. The model is suited to
  flagging cases for review, not to confirming a diagnosis.
- Model scores are not calibrated probabilities.
- Grad-CAM highlights influential regions but does not prove the anatomical location of a disease.
- The specialty table used by the application has not been reviewed by clinicians.
- The system has not been validated on clinical data and must not be used for medical diagnosis.

## 14. Team

| Name | Role |
|---|---|
| `<ENGY HESHAM HAMED>` | 
| `<JOHN GEORGE MIKHAEL>` | 
| `<MARYHAN SABRY MOUSA>` | 
| `<MENNA MOHAMED>` | 
| `<MARWA MOHAMED>` | 

