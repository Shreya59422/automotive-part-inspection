# Automotive Part Inspection System
### AI-Assisted Industrial Anomaly Detection, Visualization and Human Verification

An AI-assisted industrial inspection application built with **Streamlit**, **PatchCore**, and **Anomalib**. It supports visual anomaly detection, anomaly localization, heatmap visualization, human verification, inspection history, and an interactive 3D inspection view.

**Live application:** https://automotive-part-inspection-dashboard.streamlit.app  
**GitHub repository:** https://github.com/Shreya59422/automotive-part-inspection  
**Private model repository:** https://huggingface.co/Shreyyy7026/automotive-patchcore-model

---

## 1. Project Overview

Quality inspection is an important process in manufacturing. Manual inspection can be time-consuming and may lead to inconsistent identification of defective components.

The Automotive Part Inspection System is designed to assist visual inspection by identifying anomalous patterns in industrial component images. It combines automated anomaly detection with visual evidence and human verification.

### Objectives

- Detect anomalous patterns in industrial component images.
- Highlight potentially defective regions.
- Generate anomaly heatmaps and image overlays.
- Support human verification of AI-generated results.
- Maintain inspection records and decision history.
- Present inspection information through an interactive visualization interface.

---

## 2. Key Features

### Dashboard
- Overview of inspection activities.
- Summary of inspection records.
- Inspection statistics and visual summaries.

### AI Vision Inspection
- Upload JPG, JPEG, and PNG images.
- Run PatchCore anomaly detection.
- Generate image-level anomaly scores.
- Display anomaly heatmaps and overlays.
- Present model prediction information.

### Human Verification
- Review AI-generated inspection results.
- Manually verify component checkpoints P1–P4.
- Record operator and part identification details.
- Add inspection comments.
- Record final inspection decisions.

Verification statuses include:
- Not Verified
- Present
- Missing
- Needs Review

### XR Inspection View
- Display the original inspection image.
- Visualize anomaly heatmaps and overlays.
- Display an interactive 3D component model.
- Present P1–P4 inspection checkpoint statuses.

### Inspection History
- Store inspection records using SQLite.
- View previous inspection results.
- Display inspection summaries and decision distributions.
- Export inspection records as CSV.

---

## 3. System Architecture

```text
                         USER
                           |
                           v
                  STREAMLIT APPLICATION
                           |
             +-------------+-------------+
             |                           |
             v                           v
        IMAGE UPLOAD                 DASHBOARD
             |
             v
       IMAGE PROCESSING
             |
             v
        PATCHCORE MODEL
             |
             v
       ANOMALY DETECTION
             |
       +-----+------+
       |            |
       v            v
  ANOMALY SCORE  ANOMALY MAP
       |            |
       +-----+------+
             |
             v
        VISUALIZATION
       (Heatmap / Overlay)
             |
             v
      HUMAN VERIFICATION
             |
       +-----+------+
       |            |
       v            v
  SQLITE DATABASE  XR VIEW
       |           (3D MODEL)
       v
 INSPECTION HISTORY
       |
       v
    CSV EXPORT
```

---

## 4. Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Anomaly Detection | PatchCore |
| Deep Learning | PyTorch |
| Model Framework | Anomalib |
| Model Hosting | Hugging Face Hub |
| Image Processing | OpenCV, Pillow |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Visualization | Plotly |
| Database | SQLite |
| 3D Visualization | GLB, HTML |
| Development Environment | Visual Studio Code |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

---

## 5. Anomaly Detection Methodology

### PatchCore

PatchCore is an industrial anomaly detection approach that identifies deviations from normal image patterns. It uses pretrained feature representations and a memory bank of normal image features to identify anomalous regions.

### Workflow

1. Load the trained PatchCore checkpoint.
2. Upload an industrial component image.
3. Process the image for model inference.
4. Extract image features.
5. Compare extracted features against the learned normal feature memory bank.
6. Generate an anomaly score and anomaly map.
7. Visualize potentially anomalous regions using heatmaps and overlays.
8. Allow human verification of the prediction.

### Dataset

The model was trained and evaluated using the **MVTec Anomaly Detection (MVTec AD) dataset**, specifically its **screw category**.

The dataset provides normal training images and test images containing normal and anomalous samples.

> **Scope note:** The current model is based on the MVTec AD screw category. It has not been independently validated as an automotive-specific missing-component detection model.

---

## 6. Model Evaluation

The following metrics were obtained during the model evaluation:

| Evaluation Metric | Score |
|---|---:|
| Image AUROC | 1.0000 |
| Image F1 Score | 0.9920 |
| Pixel AUROC | 0.985593 |
| Pixel F1 Score | 0.727074 |

### Metric Interpretation

- **Image AUROC:** Measures the model's ability to distinguish normal and anomalous images.
- **Image F1 Score:** Measures the balance between precision and recall at image level.
- **Pixel AUROC:** Evaluates anomaly localization performance at pixel level.
- **Pixel F1 Score:** Evaluates pixel-level precision and recall.

These results represent the evaluated MVTec AD screw-category setup. They should not be interpreted as automotive production performance or automotive-specific validation results.

---

## 7. Human-in-the-Loop Verification

Automated anomaly detection is an inspection aid, not an independent final authority.

The human verification module allows operators to review AI-generated results and record their decisions. It supports:

- Manual component verification.
- P1–P4 checkpoint status updates.
- Operator identification.
- Part identification.
- Inspection comments.
- Final inspection decisions.

Uncertain predictions should be reviewed by a human operator.

---

## 8. Project Structure

```text
automotive-part-inspection/
│
├── app.py
│
├── backend/
│   ├── __init__.py
│   ├── database.py
│   ├── image_processing.py
│   ├── model_config.py
│   └── patchcore_service.py
│
├── generated_models/
│   └── brake_caliper.glb
│
├── patchcore_src/
│   ├── README.md
│   ├── MODEL_DOCUMENTATION.md
│   └── requirements.txt
│
├── xr/
│   └── bracket_viewer.html
│
├── requirements.txt
├── .gitignore
└── README.md
```

Datasets, model weights, virtual environments, generated inspection outputs, and local database files should not be committed to Git.

---

## 9. Local Installation and Setup

### Prerequisites

- Git
- Python version compatible with the pinned project dependencies
- Visual Studio Code (recommended)
- Access to the trained PatchCore checkpoint

### Step 1: Clone the repository

```bash
git clone https://github.com/Shreya59422/automotive-part-inspection.git
cd automotive-part-inspection
```

### Step 2: Create a virtual environment

On Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If Python 3.13 is not installed, use an installed Python version supported by the project's dependencies.

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CPU-only environment, install compatible CPU builds of PyTorch and Torchvision using the official PyTorch installation selector.

Check dependency consistency:

```bash
python -m pip check
```

---

## 10. Model Checkpoint Configuration

The trained checkpoint is approximately 261 MB and is **not stored in this GitHub repository**.

The application supports two checkpoint sources:

1. A local checkpoint for development.
2. A private Hugging Face repository for deployment or environments without a local checkpoint.

### Option A: Local checkpoint

Place the checkpoint at:

```text
models/patchcore/model.ckpt
```

Alternatively, set the `PATCHCORE_CHECKPOINT` environment variable in PowerShell:

```powershell
$env:PATCHCORE_CHECKPOINT = "C:\path\to\model.ckpt"
```

Replace the example path with the actual checkpoint location.

### Option B: Hugging Face Hub

The deployed application downloads the checkpoint from the private Hugging Face repository when a local checkpoint is not available.

Repository:

```text
Shreyyy7026/automotive-patchcore-model
```

Checkpoint filename:

```text
model.ckpt
```

The application uses `huggingface_hub` to download the checkpoint and cache it for subsequent use.

For local development with the private repository, set the token in PowerShell:

```powershell
$env:HF_TOKEN = "your_hugging_face_read_token"
```

Do not commit the token to Git or place it directly in Python source files.

---

## 11. Streamlit Community Cloud Deployment

The application is deployed using Streamlit Community Cloud.

### Step 1: Connect the GitHub repository

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Create or open the application.
3. Select the GitHub repository `Shreya59422/automotive-part-inspection`.
4. Select the `main` branch.
5. Set the application entry point to `app.py`.
6. Deploy the application.

Live app:

https://automotive-part-inspection-dashboard.streamlit.app

### Step 2: Configure the Hugging Face secret

The model repository is private, so the application needs a Hugging Face read token.

1. Create a Hugging Face **Read** access token from:
   https://huggingface.co/settings/tokens
2. Open the application settings in Streamlit Community Cloud.
3. Open **Secrets**.
4. Add the following TOML entry, replacing the placeholder with the token:

```toml
HF_TOKEN = "your_hugging_face_read_token"
```

5. Save the secret.

**Security:** Never commit the token to GitHub, include it in screenshots, or share it in public logs. If a token is exposed, revoke it and create a replacement.

### Step 3: Deploy updates

When code is pushed to the configured GitHub branch, Streamlit Community Cloud detects the changes and redeploys the application.

Typical update commands:

```powershell
git add .
git commit -m "Update application"
git push origin main
```

Only stage files that should be committed. Do not add model checkpoints, tokens, virtual environments, datasets, or local database files.

### Step 4: Verify deployment

1. Wait for the deployment to complete.
2. Open the live application.
3. Navigate to **AI Vision**.
4. Upload a supported inspection image.
5. Run an inspection and review the output.
6. If deployment or inference fails, inspect the application logs in Streamlit Community Cloud.

The first inference may take longer because the checkpoint needs to be downloaded and cached.

---

## 12. Running the Application Locally

Start the application from the project root:

```bash
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

Upload an inspection image and use the available modules for inspection, visualization, human verification, and history management.

---

## 13. Limitations and Future Scope

### Current Limitations

- The current PatchCore model is based on the MVTec AD screw category.
- Automotive-specific defect detection requires additional dataset collection and validation.
- Anomaly scores indicate model responses and are not defect probabilities.
- Anomaly heatmaps highlight relative anomaly responses and do not independently establish verified defect locations.
- P1–P4 checkpoint statuses are manually verified.
- The 3D component visualization is a schematic inspection aid, not an actual 3D reconstruction of the uploaded component.
- Human review is necessary for uncertain inspection outcomes.
- Streamlit Community Cloud local storage, including SQLite data, should not be assumed to be persistent across application restarts or redeployments.

### Future Enhancements

- Train and validate models using automotive-specific datasets.
- Extend detection to multiple automotive component categories.
- Improve anomaly localization and explainability.
- Integrate real-time camera-based inspection.
- Introduce role-based operator access.
- Enhance 3D visualization and component-level inspection.
- Evaluate deployment on industrial inspection hardware.
- Use persistent external storage for inspection records and uploaded artifacts.

---

## 14. Contributors

This project was collaboratively developed by:

1. **Shreya Reddy S**
2. **Likitha M S**
3. **Tejashwini M S**

---

## 15. Acknowledgements

We acknowledge the developers and researchers behind the MVTec AD dataset, PatchCore, PyTorch, Anomalib, Hugging Face Hub, and Streamlit for providing the resources and frameworks used in this project.

---

**Project:** Automotive Part Inspection System  
**Domain:** Artificial Intelligence | Computer Vision | Industrial Quality Inspection  
**Primary Model:** PatchCore  
**Application Framework:** Streamlit  
**Model Hosting:** Hugging Face Hub  
**Database:** SQLite
