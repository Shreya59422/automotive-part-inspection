# Automotive Part Inspection System
### Multi-Category Industrial Defect Detection and Localization Using CNN-Based PatchCore

An AI-assisted industrial inspection application designed to identify potentially defective automotive components, localize anomalous regions, and support human verification through an interactive Streamlit dashboard.

**Live Application:** https://automotive-part-inspection-dashboard.streamlit.app/

**GitHub Repository:** https://github.com/Shreya59422/automotive-part-inspection

---

## 1. Project Overview

Traditional industrial quality inspection can be time-consuming and dependent on manual examination. This project explores an automated, vision-based inspection approach using deep learning and anomaly detection.

The system uses **PatchCore**, a memory-bank-based anomaly detection approach, to identify deviations from normal visual patterns. It generates anomaly scores and heatmaps to help operators understand potentially defective regions.

The application also includes human verification to support decisions in uncertain cases.

## 2. Key Features

- **AI-Based Inspection:** Analyze uploaded industrial component images using the PatchCore model.
- **Anomaly Detection:** Identify potentially abnormal visual patterns.
- **Defect Localization:** Generate anomaly heatmaps highlighting suspicious regions.
- **Visual Evidence:** Display image overlays to support inspection decisions.
- **Human Verification:** Allow uncertain predictions to be reviewed manually.
- **Inspection History:** View previously saved inspection records.
- **Inspection Summary:** Track total inspections, accepted, rejected, and manual-review cases.
- **Result Export:** Support exporting inspection records.
- **Interactive Dashboard:** Navigate between inspection modules through a Streamlit interface.

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Anomaly Detection | PatchCore |
| Computer Vision | OpenCV |
| Model Framework | Anomalib |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Web Application | Streamlit |
| Model Hosting | Hugging Face |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |
| Database | SQLite |

## 4. System Architecture

The application follows this general workflow:

1. User uploads an industrial component image.
2. The image is preprocessed.
3. The PatchCore model analyzes visual features.
4. An anomaly score is generated.
5. An anomaly heatmap and visual overlay are produced.
6. The system presents an inspection result.
7. Uncertain cases can be sent for human verification.
8. Inspection details are saved in the history.

## 5. Application Modules

| Module | Description |
|---|---|
| Dashboard | Provides an overview of inspection activities. |
| AI Vision | Performs image-based anomaly inspection. |
| Human Verification | Supports manual review of uncertain predictions. |
| XR Inspection | Provides the XR inspection interface. |
| Inspection History | Displays saved inspection records and summary statistics. |

## 6. Model Information

**Primary Model:** PatchCore

PatchCore is an anomaly detection method that learns representative feature patterns from normal training images and identifies deviations during inference.

**Training Dataset:** MVTec Anomaly Detection (MVTec AD), screw category.

The model was trained using normal training images. Defective test images were not used for training.

**Important:** The current model is trained on the MVTec AD screw category. Its performance on other automotive component categories requires separate dataset validation and testing.

Anomaly scores are not calibrated defect probabilities. Predictions should be interpreted alongside visual evidence and human verification.

## 7. Model Hosting

The trained checkpoint is hosted in a private Hugging Face repository.

**Model Repository:**  
https://huggingface.co/Shreyyy7026/automotive-patchcore-model

The application retrieves the model checkpoint using a Hugging Face access token.

Configure the following secret in Streamlit Community Cloud:

```toml
HF_TOKEN = "your_huggingface_read_token"
```

Never upload access tokens or private credentials to GitHub.

## 8. Project Structure

```text
automotive-part-inspection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── backend/
│   ├── __init__.py
│   ├── database.py
│   ├── image_processing.py
│   ├── model_config.py
│   └── patchcore_service.py
│
└── .gitignore
```

## 9. Installation and Local Setup

### Prerequisites

- Python 3.12
- Git
- Visual Studio Code (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Shreya59422/automotive-part-inspection.git
```

### Step 2: Navigate to the Project

```bash
cd automotive-part-inspection
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Configure Hugging Face Access

Set your Hugging Face read token securely as an environment variable or through the application's supported configuration.

### Step 7: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 10. Deployment

The application is deployed using Streamlit Community Cloud.

**Deployment Configuration**

- Repository: `Shreya59422/automotive-part-inspection`
- Branch: `main`
- Main file: `app.py`
- Python version: `3.12`
- Model hosting: Hugging Face

**Live Application:**  
https://automotive-part-inspection-dashboard.streamlit.app/

## 11. Performance Evaluation

The PatchCore model was evaluated on the MVTec AD screw category.

| Evaluation Metric | Score |
|---|---:|
| Image AUROC | 1.0000 |
| Image F1-Score | 0.9920 |
| Pixel AUROC | 0.985593 |
| Pixel F1-Score | 0.727074 |

These results describe the evaluated MVTec screw dataset and should not be interpreted as validated performance across all automotive components.

## 12. Future Enhancements

- Expand training and evaluation to multiple automotive component categories.
- Improve anomaly localization and segmentation performance.
- Introduce additional defect classification capabilities.
- Integrate real-time camera-based inspection.
- Enhance explainability and evidence visualization.
- Improve human-in-the-loop review workflows.
- Add persistent cloud-based inspection storage.
- Explore XR-assisted industrial inspection.

## 13. Contributors

**Shreya Reddy S**
**Likitha M S**
**Tejashwini S**
---

*This project demonstrates the application of deep learning, computer vision, anomaly detection, and human-in-the-loop verification for industrial quality inspection.*