# Multi-Category Industrial Defect Detection and Localization Using CNN-Based PatchCore

An industrial visual inspection system that detects anomalous products and localizes potential defects using PatchCore with a pretrained CNN feature extractor. The project is evaluated on all 15 categories of the MVTec AD dataset.

**Live Demo:** https://cnn-defect-detection-manufacturingprocess.streamlit.app/  
**GitHub Repository:** https://github.com/Shreya59422/CNN-Defect-Detection

## Project Overview

Quality inspection is an important part of manufacturing. Traditional manual inspection can be time-consuming and may lead to inconsistent results.

This project implements a deep-learning-based industrial anomaly detection system that learns the appearance of normal products and identifies deviations that may indicate defects.

The system uses PatchCore with a pretrained WideResNet50_2 CNN backbone to extract visual features, build a memory bank of normal features, and calculate anomaly scores for inspection images.

It also generates anomaly maps to highlight suspicious regions.

## Objectives

- Detect defective and anomalous industrial products.
- Identify potential defect locations using anomaly maps.
- Evaluate image-level and pixel-level detection performance.
- Support multiple industrial product categories.
- Provide an interactive interface for image inspection.
- Explore the use of CNN-based feature extraction for automated quality control.

## Dataset

**MVTec Anomaly Detection (MVTec AD)**

The dataset contains normal and defective images from 15 industrial object and texture categories.

The project covers all 15 categories:

| Object Categories | Texture Categories |
|---|---|
| Bottle | Carpet |
| Cable | Grid |
| Capsule | Leather |
| Hazelnut | Tile |
| Metal Nut | Wood |
| Pill | |
| Screw | |
| Toothbrush | |
| Transistor | |
| Zipper | |

**Training approach:** Normal images are used to build the PatchCore memory bank. Defective images are used for evaluation, not for training the normal reference memory.

## Methodology

### 1. Image Preprocessing
- Load and prepare industrial inspection images.
- Apply the preprocessing required by the pretrained feature extractor.
- Resize and normalize images for model input.

### 2. CNN Feature Extraction
- Use pretrained WideResNet50_2.
- Extract intermediate feature representations from `layer2` and `layer3`.
- Combine visual information from different feature levels.

### 3. PatchCore Anomaly Detection
- Build a memory bank using normal training image features.
- Apply coreset subsampling to reduce memory requirements.
- Compare test-image features with the stored normal feature representations.
- Calculate anomaly scores based on nearest-neighbor distances.

### 4. Anomaly Localization
- Generate pixel-level anomaly maps.
- Visualize suspicious regions using heatmaps and overlays.
- Compare predicted anomaly maps with available ground-truth masks during evaluation.

### 5. Performance Evaluation
Evaluate the system using:
- Image AUROC
- Image F1-score
- Pixel AUROC
- Pixel F1-score

## Model Configuration

| Component | Configuration |
|---|---|
| Anomaly detection model | PatchCore |
| CNN backbone | WideResNet50_2 |
| Feature extraction layers | layer2, layer3 |
| Coreset sampling ratio | 0.1 |
| Nearest neighbors | 9 |
| Training data | Normal images |
| Dataset | MVTec AD |
| Categories | 15 |

## Evaluation Results

The following table presents the category-wise results from `patchcore_all_categories.csv`.

| Category | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 |
|---|---:|---:|---:|---:|
| Bottle | 1.0000 | 0.9920 | 0.9856 | 0.7262 |
| Cable | 0.9818 | 0.9622 | 0.9847 | 0.6383 |
| Capsule | 0.9924 | 0.9765 | 0.9901 | 0.5157 |
| Carpet | 0.9876 | 0.9773 | 0.9906 | 0.6048 |
| Grid | 0.9883 | 0.9565 | 0.9815 | 0.3852 |
| Hazelnut | 1.0000 | 0.9928 | 0.9883 | 0.6299 |
| Leather | 1.0000 | 0.9945 | 0.9922 | 0.4406 |
| Metal Nut | 0.9980 | 0.9840 | 0.9870 | 0.8406 |
| Pill | 0.9449 | 0.9489 | 0.9809 | 0.7193 |
| Screw | 0.9496 | 0.9328 | 0.9887 | 0.3735 |
| Tile | 0.9996 | 0.9881 | 0.9554 | 0.6193 |
| Toothbrush | 0.9167 | 0.9355 | 0.9888 | 0.5951 |
| Transistor | 0.9946 | 0.9620 | 0.9735 | 0.6133 |
| Wood | 0.9868 | 0.9661 | 0.9315 | 0.4716 |
| Zipper | 0.9769 | 0.9789 | 0.9814 | 0.5440 |

**Note:** Image-level metrics measure the ability to distinguish normal from anomalous images. Pixel-level metrics measure defect localization. A high pixel AUROC does not necessarily imply a high pixel F1-score; localization quality varies across categories.

## Streamlit Application

The project includes an interactive Streamlit interface for visual inspection.

Features include:
- Image upload.
- Anomaly prediction.
- Anomaly heatmap visualization.
- Original image and heatmap comparison.
- Model performance information.
- Inspection history during the application session.

**Live Application:** https://cnn-defect-detection-manufacturingprocess.streamlit.app/

Note: The currently deployed application uses the screw-category checkpoint. The project has checkpoints and evaluation results for all 15 categories, but the dashboard should not be described as supporting all-category selection until that functionality is implemented and tested.

## Technologies Used

- Python
- PyTorch
- Anomalib
- PatchCore
- WideResNet50_2
- Streamlit
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn

## Project Structure

```text
CNN-Defect-Detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── results/
│   ├── anomaly_detection/
│   │   └── Patchcore/
│   │       └── MVTecAD/
│   │           ├── bottle/
│   │           ├── cable/
│   │           ├── capsule/
│   │           ├── carpet/
│   │           ├── grid/
│   │           ├── hazelnut/
│   │           ├── leather/
│   │           ├── metal_nut/
│   │           ├── pill/
│   │           ├── screw/
│   │           ├── tile/
│   │           ├── toothbrush/
│   │           ├── transistor/
│   │           ├── wood/
│   │           └── zipper/
│   │
│   ├── patchcore_all_categories.csv
│   ├── patchcore_category_comparison.csv
│   └── patchcore_category_metrics_ranked.csv
│
└── scripts/
    └── training and evaluation scripts
```

The directory listing is representative. Large model checkpoints and datasets may be stored separately rather than committed to GitHub.

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/Shreya59422/CNN-Defect-Detection.git
cd CNN-Defect-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

**Important:** The required model checkpoint must be available at the path expected by the application. Checkpoints are large files and may not be included in the Git repository.

## Limitations

- Pixel-level localization performance varies across product categories.
- The system relies on pretrained CNN feature representations.
- The current Streamlit deployment is configured for the screw category.
- Robustness under different lighting and imaging conditions has not yet been systematically evaluated.
- Predictions should be reviewed by a human inspector before being used for critical manufacturing decisions.

## Future Enhancements

- Add category selection to the Streamlit dashboard.
- Improve pixel-level defect localization.
- Introduce a human-in-the-loop review workflow for uncertain predictions.
- Evaluate robustness under different lighting and image acquisition conditions.
- Optimize inference speed and model memory usage.
- Explore comparisons with additional anomaly detection architectures.

## Author
**Shreya Reddy S**
**Likitha M S**
**Tejashwini S**


**GitHub:** https://github.com/Shreya59422
            https://github.com/likithams09
            https://github.com/teju123540
