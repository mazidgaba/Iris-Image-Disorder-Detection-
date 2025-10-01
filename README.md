## Overview

This repository contains a Jupyter Notebook implementation of a simple AI prototype for detecting eye disorders (Healthy vs. Diseased iris) using transfer learning with MobileNetV2. The project demonstrates image preprocessing, model training, evaluation, and explainability via Grad-CAM heatmaps. It was developed as an assignment for an AI/ML Intern position at Dr. ViKi, focusing on clarity and transparency rather than high accuracy.

Key highlights:
- **Dataset**: Small subset (200 images: 100 Healthy, 100 Diseased) from a public iris dataset (e.g., Kaggle Iris Disease Dataset).
- **Model**: Pre-trained MobileNetV2 (frozen base) fine-tuned for binary classification.
- **Results**: Test accuracy ~72.5% (on small dataset); Grad-CAM visualizations show model focus on iris regions.
- **Tools**: TensorFlow/Keras, Matplotlib, Scikit-learn, PIL.

The pipeline ensures no patient-identifiable data is used.

## Objectives

- Load and visualize iris images.
- Preprocess (resize 224x224, normalize, 80/20 split with augmentation).
- Build and train a transfer learning model (3 epochs).
- Evaluate with accuracy, confusion matrix, and loss/accuracy plots.
- Generate Grad-CAM heatmaps for explainability.

## Dataset

- **Source**: Public Iris Disease Dataset (Kaggle) or similar (e.g., CASIA subset). Download a small folder structure: `Healthy/` and `Diseased/` with ~100 images each.
- **Structure**:
  ```
  irisdata/
  ├── Healthy/
  │   ├── img1.jpg
  │   └── ...
  └── Diseased/
      ├── img1.jpg
      └── ...
  ```
- **Size**: Kept small for quick runtime (~23s training on CPU).

No internet access needed post-download; anonymized iris crops only.

## Installation

1. **Clone the Repo**:
   ```
   git clone https://github.com/yourusername/iris-disorder-detection.git
   cd iris-disorder-detection
   ```

2. **Environment Setup** (Recommended: Google Colab with GPU runtime):
   - Create a virtual environment:
     ```
     python -m venv iris_env
     source iris_env/bin/activate  # On Windows: iris_env\Scripts\activate
     ```
   - Install dependencies:
     ```
     pip install tensorflow==2.19.0 numpy matplotlib pillow scikit-learn
     ```
   - Or use `requirements.txt` (if provided):
     ```
     pip install -r requirements.txt
     ```

3. **Download Dataset**:
   - Place images in `irisdata/` folder as described above.
   - Update `data_dir` path in the notebook if needed.

## Usage

1. **Run the Notebook**:
   - Open `iris_disorder_detection.ipynb` in Jupyter Notebook or Google Colab.
   - Execute cells sequentially:
     - Cell 1: Environment setup (prints Python/TF versions, GPU check).
     - Cell 2: Data loading & visualization (samples grid).
     - Cell 3: Preprocessing (dataset split, augmentation, post-process viz).
     - Cell 4: Model building & training (MobileNetV2, 3 epochs).
     - Cell 5: Evaluation (accuracy, confusion matrix, plots).
     - Cell 6: Grad-CAM (generates 3 overlay PNGs in `gradcam_output/`).
     - Final: Summary & visualizations.

2. **Expected Outputs**:
   - Sample images: 2x3 grid.
   - Preprocessed samples: 3x3 grid.
   - Model summary: ~2.26M params.
   - Training: Progress bars with loss/accuracy.
   - Evaluation: Prints accuracy (e.g., 0.7250), confusion matrix, report; loss/acc plots.
   - Grad-CAM: 3 PNG overlays (e.g., `gradcam_1_TrueDiseased_PredDiseased.png`); inline display.
   - Summary: Key findings paragraph.

**Runtime**: ~1-2 min on CPU; faster on GPU.

**Troubleshooting**:
- Deprecation warnings (NumPy/Sklearn): Benign; suppress with `import warnings; warnings.filterwarnings('ignore')`.
- Gradients None (frozen model): Produces noisy heatmaps; unfreeze base for better results.
- No GPU: Set `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'` to reduce logs.

## Project Structure

```
iris-disorder-detection/
├── iris_disorder_detection.ipynb      # Main notebook
├── irisdata/                          # Dataset folder
│   ├── Healthy/
│   └── Diseased/
├── gradcam_output/                    # Generated heatmaps (auto-created)
├── requirements.txt                   # Dependencies
└── README.md                          # This file
```

## Results & Findings

- **Accuracy**: 72.5% on test set (40 images; imbalance noted in confusion matrix: all Healthy misclassified).
- **Training Curves**: Converging loss (train: 0.96 → 0.44; val: 0.92 → 0.55); accuracy improves (train: 47% → 82%; val: 23% → 73%).
- **Grad-CAM**: Heatmaps highlight central iris regions (noisy due to frozen layers, but functional).
- **Insights**: Model favors Diseased class; small dataset limits generalization.

See notebook for plots and outputs.

## Future Improvements

- Larger/diverse dataset for balanced splits.
- Fine-tune base model after warm-up.
- Domain-specific preprocessing (e.g., iris segmentation via OpenCV).
- Advanced explainability (e.g., SHAP) or deployment (Streamlit/Flask app).
- Multi-class support for specific disorders.

## Contributing

Fork the repo, create a branch, and submit a PR. Issues welcome!

## Contact

Gulam Mazid - mazidgulam786@gmail.com  
