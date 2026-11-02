# Automotive Part Inspection System
### AI-Assisted Industrial Anomaly Detection, Visualization and Human Verification

An AI-assisted industrial inspection application that uses **PatchCore** for visual anomaly detection, defect localization, and heatmap generation. The system integrates human verification, inspection history, and an interactive 3D visualization interface to support industrial quality inspection workflows.

---

## 1. Project Overview

Quality inspection is an important process in manufacturing industries. Traditional manual inspection can be time-consuming and may lead to inconsistent identification of defective components.

The **Automotive Part Inspection System** is designed to support visual inspection by identifying anomalous patterns in industrial component images using deep learning.

The application combines automated anomaly detection with visual evidence and human verification to provide a structured inspection workflow.

### Objectives

- Detect anomalous patterns in industrial component images.
- Identify and visualize potentially defective regions.
- Generate anomaly heatmaps and image overlays.
- Provide an interface for human verification.
- Maintain inspection records and decision history.
- Present inspection information through an interactive visualization interface.

---

## 2. Key Features

### 2.1 Dashboard
- Overview of inspection activities.
- Summary of inspection records.
- Inspection statistics and visual summaries.

### 2.2 AI Vision Inspection
- Upload images in JPG, JPEG, and PNG formats.
- Perform anomaly detection using PatchCore.
- Generate anomaly scores.
- Display anomaly heatmaps.
- Generate image overlays for visual interpretation.
- Display model prediction information.

### 2.3 Human Verification
- Review AI-generated inspection results.
- Manually verify component checkpoints P1–P4.
- Record operator information and part identification.
- Add inspection comments.
- Record final inspection decisions.

Verification statuses:
- Not Verified
- Present
- Missing
- Needs Review

### 2.4 XR Inspection View
- Display the original inspection image.
- Visualize anomaly heatmaps and overlays.
- Display an interactive 3D component model.
- Present P1–P4 inspection checkpoint statuses.

### 2.5 Inspection History
- Store inspection records using SQLite.
- View previous inspection results.
- Display inspection summaries.
- Analyze decision distributions.
- Export inspection records as CSV.

---

## 3. System Architecture

The application follows a modular architecture that separates the user interface, image processing, anomaly detection, database operations, and visualization components.

```text
                 USER
                   |
                   v
          STREAMLIT APPLICATION
                   |
          +--------+---------+
          |                  |
          v                  v
     IMAGE UPLOAD       DASHBOARD
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
     +----+-----+
     |          |
     v          v
 ANOMALY     ANOMALY
  SCORE       MAP
     |          |
     +----+-----+
          |
          v
   VISUALIZATION
  (Heatmap / Overlay)
          |
          v
   HUMAN VERIFICATION
          |
     +----+------+
     |           |
     v           v
  SQLITE       XR VIEW
  DATABASE     (3D MODEL)
     |
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
| Image Processing | OpenCV, Pillow |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Visualization | Plotly |
| Database | SQLite |
| 3D Visualization | GLB, HTML |
| Development Environment | Visual Studio Code |
| Version Control | Git, GitHub |

---

## 5. Anomaly Detection Methodology

### PatchCore

PatchCore is an industrial anomaly detection approach that identifies deviations from normal image patterns.

It uses pretrained feature representations and a memory bank of normal image features to identify anomalous regions.

The model is particularly suitable for anomaly detection scenarios where defective training samples are limited or unavailable.

### Workflow

1. Load the trained PatchCore checkpoint.
2. Upload an industrial component image.
3. Process the image for model inference.
4. Extract image features.
5. Compare extracted features against the learned normal feature memory bank.
6. Generate an anomaly score.
7. Generate an anomaly map.
8. Visualize potentially anomalous regions using heatmaps and overlays.
9. Allow human verification of the prediction.

### Dataset

The model was trained and evaluated using the **MVTec Anomaly Detection (MVTec AD) dataset**, specifically its screw category.

The dataset provides normal training images and test images containing normal and anomalous samples.

**Important:** The current model is based on the MVTec AD screw category. It has not been independently validated as an automotive-specific missing-component detection model.

---

## 6. Model Evaluation

The following metrics were obtained during model evaluation:

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

Automated anomaly detection is used as an inspection aid rather than an independent final authority.

The human verification module allows operators to review AI-generated results and record their decisions.

The workflow supports:

- Manual component verification.
- P1–P4 checkpoint status updates.
- Operator identification.
- Part identification.
- Inspection comments.
- Final inspection decisions.

This approach helps ensure that uncertain predictions can be reviewed by a human operator.

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

Additional local folders such as datasets, model weights, virtual environments, generated inspection outputs, and the SQLite database are excluded from version control.

---

## 9. Installation and Setup

### Prerequisites

- Python 3.13
- Git
- Visual Studio Code
- A trained PatchCore checkpoint

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd automotive-part-inspection
```

### Step 2: Create a Virtual Environment

```bash
py -3.13 -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CPU-based PyTorch installation, if required:

```bash
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```

Check the installed dependencies:

```bash
python -m pip check
```

---

## 10. Model Checkpoint Configuration

The trained PatchCore checkpoint is not included in this GitHub repository because of its large file size.

Obtain the checkpoint separately and configure its location using the `PATCHCORE_CHECKPOINT` environment variable.

### Windows PowerShell

```powershell
$env:PATCHCORE_CHECKPOINT = "C:\path\to\model.ckpt"
```

Replace the example path with the actual checkpoint location.

The application also supports a default checkpoint location:

```text
models/patchcore/model.ckpt
```

This location is defined in `backend/model_config.py`.

### Verify the Checkpoint

```powershell
python -c "from backend.model_config import PATCHCORE_CHECKPOINT; print(PATCHCORE_CHECKPOINT); print(PATCHCORE_CHECKPOINT.exists())"
```

The final output should display `True` if the checkpoint exists.

---

## 11. Running the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

Upload an inspection image and use the available application modules to perform inspection, visualization, human verification, and history management.

---

## 12. Limitations and Future Scope

### Current Limitations

- The current PatchCore model is based on the MVTec AD screw category.
- Automotive-specific defect detection requires additional dataset collection and validation.
- Anomaly scores indicate model responses and are not defect probabilities.
- Anomaly heatmaps highlight relative anomaly responses and do not independently establish verified defect locations.
- P1–P4 checkpoint statuses are manually verified.
- The 3D component visualization is a schematic inspection aid, not an actual 3D reconstruction of the uploaded component.
- Human review is necessary for uncertain inspection outcomes.

### Future Enhancements

- Train and validate models using automotive-specific datasets.
- Extend detection to multiple automotive component categories.
- Improve anomaly localization and explainability.
- Integrate real-time camera-based inspection.
- Introduce role-based operator access.
- Enhance 3D visualization and component-level inspection.
- Evaluate deployment on industrial inspection hardware.

---

## 13. Contributors

This project was collaboratively developed by:

1. **Shreya Reddy S**
2. **Likitha M S**
3. **Tejashwini M S**

---

## 14. Acknowledgements

We acknowledge the developers and researchers behind the MVTec AD dataset, PatchCore, PyTorch, and Anomalib for providing the resources and frameworks used in this project.

---

**Project:** Automotive Part Inspection System  
**Domain:** Artificial Intelligence | Computer Vision | Industrial Quality Inspection  
**Primary Model:** PatchCore  
**Application Framework:** Streamlit  
**Database:** SQLite
