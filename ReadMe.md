# Neurolight
## Sample code for : Sex Classification Project
This repository contains code for sex classification using deep learning models implemented with PyTorch. It includes various scripts that demonstrate the entire process, from basic implementation to advanced features like cross-validation and distributed training.

## Project Structure

The project consists of the following key components:

### 1. **sex_classification.ipynb** 
   - This notebook is a standalone script that follows a tutorial on sex classification.
   - It walks through the basic steps of building a neural network model for sex classification using PyTorch.
   - You can find an explanation of this notebook in detail in the [Medium Article](https://medium.com/@daminininad/intro-to-ai-with-neuroimaging-data-a-end-to-end-tutorial-using-pytorch-f941c6ef547a).
   - This notebook provides an introduction to the model, dataset, and the general structure of the classification task.

### 2. **crossvalidation_sexclassification.py** 
   - This script builds upon the ideas from the notebook and implements K-Fold Cross Validation for evaluating the model's performance.
   - It is a standalone script and can be executed independently.
   - The script uses PyTorch for model training and validation with K-folds to enhance the model’s generalization.

### 3. **ddp_sex.py** (Distributed Data Parallel Implementation)
   - This is the most crucial script of the project, written from scratch to implement Distributed Data Parallel (DDP) for efficient training on multiple GPUs.
   - It should be run using the command:
     ```bash
     torchrun --standalone --nproc_per_node=7 ddp_sex.py
     ```
   - The script requires specific folder structures and is not standalone; it depends on other components in the project. Please make sure the necessary directories and files are available before running the script.

### 4. **/data** Folder
   - **dataset.py**: Defines the dataset class for loading and processing the data.
   - **dataloader.py**: Handles the data loading process, including batching, shuffling, and creating data loaders for training.

### 5. **/utils** Folder
   - **trainer.py**: Implements the training loop, including the forward and backward passes, as well as model evaluation.
   - **visualisation.py**: Provides functions for visualizing results like training loss curves, confusion matrices, and other metrics.

### 6. **/models** Folder
   - Contains the implementations of different models for sex classification, including:
     - **SFCN** (Shallow Fully Connected Network)
     - **ResNet**: A deep residual network, commonly used for image classification tasks.

### 7. **/configs** Folder
   - **config.yaml**: Stores configuration parameters like learning rates, batch sizes, and model-specific parameters. It makes the scripts more flexible by allowing easy modification of settings.

### 8. **requirements.txt**
   - Lists the required Python libraries and dependencies for the project. Install the required packages using:
     ```bash
     pip install -r requirements.txt
     ```

## How to Run the Code 🏃

### Step 1: Clone the Repository
```bash
git clone <repository_url>
cd <repository_name>
```
### Step 2: Install Dependencies

Make sure to install the necessary Python dependencies:
```bash
pip install -r requirements.txt

```
### Step 3: Run the Scripts
sex_classification.ipynb: Open this notebook and run the cells to get started with sex classification.
```python3
    python3 crossvalidation_sexclassification.py #Run this script for K-fold cross-validation.

```

ddp_sex.py: For distributed training, run this script with the following command:
```python3
  torchrun --standalone --nproc_per_node=7 ddp_sex.py
```
    

## Folder Overview

    /data: Contains the dataset and data processing utilities.
    /utils: Contains utility functions for training and visualization.
    /models: Defines the model architectures used in this project.
    /configs: Configuration files for adjusting hyperparameters and model settings.

## Requirements

Make sure to install the required dependencies from requirements.txt.
```bash
  pip install -r requirements.txt
```


## Conclusion

This repository provides a comprehensive pipeline for sex classification using deep learning. It includes a simple tutorial-based approach, an extended K-fold cross-validation script, and a more advanced distributed training script leveraging Distributed Data Parallel in PyTorch for scaling the training process across multiple GPUs.


