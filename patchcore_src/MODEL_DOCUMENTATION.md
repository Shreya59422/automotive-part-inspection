# PatchCore Source Code – Technical Documentation

## 1. Project Overview

This folder contains the preserved source code from the original CNN Defect Detection project.

The original project investigates industrial anomaly detection using CNN-based methods and PatchCore on the MVTec AD dataset.

The source files are preserved for inspection, explanation and integration into the Automotive Part Inspection project.

## 2. Technologies Used

- Python
- PyTorch
- Torchvision
- Anomalib
- PatchCore
- WideResNet50_2 feature extractor
- NumPy
- OpenCV
- Matplotlib
- Streamlit
- Pandas
- Scikit-learn

Refer to requirements.txt for the original project dependencies.

## 3. Dataset

The original model uses the MVTec Anomaly Detection (MVTec AD) dataset.

The dataset contains categories such as:

- Bottle
- Cable
- Capsule
- Carpet
- Grid
- Hazelnut
- Leather
- Metal Nut
- Pill
- Screw
- Tile
- Toothbrush
- Transistor
- Wood
- Zipper

The screw category contains normal training images and test images with normal and defective examples.

## 4. PatchCore Architecture

PatchCore is an industrial visual anomaly detection method.

The implementation uses a pretrained WideResNet50_2 CNN feature extractor.

The existing configuration uses:
- Feature extraction layers: layer2 and layer3
- Coreset sampling ratio: 0.1
- Nearest-neighbour setting: 9

### Main processing stages

1. Load an input image.
2. Extract visual features using the CNN backbone.
3. Generate patch-level feature representations.
4. Build a representative memory bank from normal training features.
5. Compare test image features with the stored normal feature representations.
6. Calculate anomaly scores.
7. Generate anomaly maps for visual localization.

## 5. Training and Experiment Scripts

The src folder contains the original Python implementation.

Important scripts include:

- 1_inspect_mvtec.py – dataset inspection.
- 2_train_cnn.py – CNN training experiment.
- 3_train_patchcore.py – PatchCore training.
- 4_generate_masks.py – mask generation.
- 5_evaluate_patchcore.py – PatchCore evaluation.
- 6_threshold_evaluation.py – threshold evaluation.
- 10_plot_category_results.py – category result visualization.
- 22_train_all_patchcore.py – multi-category PatchCore training.
- 23_evaluate_metrics.py – metric evaluation.
- 24_evaluate_all_patchcore.py – multi-category evaluation.
- 27_visualize_screw_prediction.py – screw prediction visualization.
- 28_check_localization.py – localization checks.
- 29_visualize_localization.py – localization visualization.
- 30_train_screw_patchcore.py – screw-specific training experiment.
- 31_compare_defect_scores.py – defect score comparison.
- 32_evaluate_all_defects.py – defect evaluation.
- 33_evaluate_patchcore.py – PatchCore evaluation.
- 34_diagnose_pixel_threshold.py – pixel threshold diagnostics.
- 35_check_validation_threshold.py – validation threshold checks.

Other scripts are preserved in src and can be inspected individually.

## 6. Model Checkpoint

The existing trained screw PatchCore checkpoint is:

results/anomaly_detection/Patchcore/MVTecAD/screw/v0/weights/lightning/model.ckpt

The automotive inspection project currently references this checkpoint through:

backend/model_config.py

The checkpoint is reused for inference. It is not retrained as part of the initial integration.

## 7. Inference Workflow

The new integration uses:

backend/patchcore_service.py

The service:
1. Receives an image path.
2. Loads the PatchCore model.
3. Loads the existing checkpoint.
4. Runs inference using the Anomalib Engine.
5. Returns the anomaly score, predicted label, anomaly map and prediction mask.

## 8. Image Evidence Visualization

The new integration uses:

backend/image_processing.py

It provides:
- Anomaly heatmap generation.
- Heatmap overlay on the original image.

These visual outputs help an operator examine areas identified as potentially abnormal.

## 9. Evaluation

The original project contains scripts for:
- Image-level evaluation.
- Pixel-level evaluation.
- AUROC and F1-score analysis.
- Threshold diagnostics.
- Defect score comparison.
- Anomaly localization visualization.

Previously recorded MVTec screw results include:

- Image AUROC: 1.000
- Image F1-score: 0.992
- Pixel AUROC: 0.985593
- Pixel F1-score: 0.727074

These are historical MVTec screw experiment results, not automotive inspection performance results.

## 10. Integration into Automotive Part Inspection

The PatchCore implementation is one component of the larger automotive inspection system.

The intended integrated workflow is:

1. Capture or upload an automotive part image.
2. Process the image using the AI vision component.
3. Generate anomaly evidence.
4. Display the result to the human operator.
5. Allow Accept, Reject or Manual Review decisions.
6. Record inspection information.

The browser-based 3D/XR interface is a separate component of the new project.

## 11. Important Limitations

- The existing checkpoint was trained using MVTec AD screw images.
- It has not been validated on automotive mounting bracket images.
- MVTec evaluation metrics must not be presented as automotive inspection metrics.
- A normal MVTec screw image has previously received an anomalous prediction.
- Anomaly scores require appropriate calibration and validation.
- Human verification is retained for uncertain inspection cases.

## 12. Source Code and Reuse

The Python source files in this folder are preserved from the original project.

The original project remains separate and unchanged.

New automotive inspection modules are developed separately in the main project.

Reused components and newly developed components should be identified clearly when presenting the project.
