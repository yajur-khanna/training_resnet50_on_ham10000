````markdown name=README.md url=https://github.com/yajur-khanna/training_resnet50_on_ham10000/blob/1241de494df7b6bee7c45e98c8a0b84aef244c5a/README.md
# Training ResNet50 on HAM10000

This repository contains a Jupyter notebook (`resnet50.ipynb`) that demonstrates training a ResNet-50 model on the HAM10000 skin lesion dataset for multi-class skin lesion classification.

The notebook includes data loading and preprocessing, augmentation, model building (ResNet50 as backbone), training, evaluation, and visualization of results.

## Table of contents

- [Repository structure](#repository-structure)
- [Notebook overview](#notebook-overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training tips & considerations](#training-tips--considerations)
- [Results & artifacts](#results--artifacts)
- [Credits & references](#credits--references)
- [License](#license)
- [Contact](#contact)

## Repository structure

- `resnet50.ipynb` — Main notebook implementing data pipeline and training.
- (Optionally) `requirements.txt` — Python dependencies (not present in repo; see [Requirements](#requirements) below).
- Model checkpoints, logs, or exported artifacts (not committed) — saved during training by the notebook.

## Notebook overview

The `resnet50.ipynb` notebook demonstrates:

1. Loading HAM10000 images and labels
2. Exploratory data analysis and class distribution checks
3. Preprocessing and augmentation pipelines
4. Constructing a ResNet-50 based classifier (transfer learning / fine-tuning)
5. Training loop with callbacks (early stopping, model checkpointing, learning rate scheduling)
6. Evaluation on a hold-out set and visualization (confusion matrix, sample predictions)
7. Saving the trained model and key metrics

Open the notebook to see the exact model configuration, hyperparameters, and training schedule used.

## Requirements

The notebook was developed for Python 3. (Notebook metadata shows Python 3.12, but any modern Python 3.8+ environment should work with appropriate package versions.)

Typical dependencies (install via pip):

```bash
pip install -r requirements.txt
```

If a `requirements.txt` is not provided, install the common packages used for training:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras jupyter seaborn pillow opencv-python tqdm
```

If you prefer Conda:

```bash
conda create -n resnet50-ham10000 python=3.9
conda activate resnet50-ham10000
pip install -r requirements.txt
```

GPU is recommended (CUDA + cuDNN) for model training performance.

## Dataset

This notebook expects the HAM10000 ("Human Against Machine with 10000 training images") dataset. The dataset can be obtained from:

- ISIC / HAM10000 releases (research dataset)
- Kaggle (search for "HAM10000")

Place the images and metadata in the paths expected by the notebook. Typical layout:

```
data/
  HAM10000_images_part_1/
  HAM10000_images_part_2/
  HAM10000_metadata.csv
```

Adjust paths in the notebook to point to your local dataset copy. The notebook contains cells that parse the metadata CSV and map image filenames to labels.

IMPORTANT: Ensure you follow the HAM10000 dataset license and citation requirements when using and distributing results.

## Usage

1. Clone this repository:

```bash
git clone https://github.com/yajur-khanna/training_resnet50_on_ham10000.git
cd training_resnet50_on_ham10000
```

2. Prepare your environment and install dependencies.

3. Download and prepare the HAM10000 dataset and update dataset paths inside `resnet50.ipynb`.

4. Launch Jupyter Notebook / JupyterLab:

```bash
jupyter notebook resnet50.ipynb
# or
jupyter lab resnet50.ipynb
```

5. Execute the notebook cells sequentially. Typical steps:
   - Data loading & preprocessing
   - Create train / validation / test splits (the notebook includes splitting code)
   - Build the ResNet50 model (transfer learning)
   - Train the model (monitor checkpoints and logs)
   - Evaluate and visualize results

If you prefer to run training from a Python script, you can convert the notebook to a script:

```bash
jupyter nbconvert --to script resnet50.ipynb
python resnet50.py
```

(You may need to edit the generated script to adapt interactive cells or magic commands.)

## Training tips & considerations

- Use a GPU-enabled machine for reasonable training time.
- Balance classes or use class weights / oversampling when handling class imbalance.
- Start with a smaller number of epochs to sanity-check the pipeline.
- Use appropriate input size consistent with ResNet50 (commonly 224x224) unless modified in the notebook.
- Consider using modern augmentation libraries (albumentations) for improved robustness.
- Monitor training with TensorBoard or use in-notebook plots.

## Results & artifacts

The notebook saves checkpoints and evaluation figures to disk (see the notebook for exact output paths). Typical artifacts:

- Model checkpoint files (.h5 / SavedModel)
- Training history (loss, accuracy plots)
- Confusion matrix and sample prediction images
- Logs for reproducibility

If you publish results, include the dataset split and random seed used to allow reproducibility.

## Credits & references

- HAM10000 dataset: Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data.
- ResNet: He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition.

## License

This repository does not contain the HAM10000 images (data must be obtained separately). Code in this repository is provided under the repository license (add a LICENSE file if needed). When using HAM10000, follow the dataset's license and citation requirements.

## Contact

Author: yajur-khanna (GitHub: [yajur-khanna](https://github.com/yajur-khanna))

If you find issues or have questions about the notebook, please open an issue in the repository.

````
