# Oxford-IIIT Pet Dataset Management with DVC, MLFlow, and TensorBoard

## Project Description

This project manages the **Oxford-IIIT Pet Dataset** using **DVC** (Data Version Control), **MLFlow** for model tracking, and **TensorBoard** for visualization. The dataset contains images of 37 different breeds of cats and dogs, and the project demonstrates how to handle large datasets, version control models, and monitor training progress in a machine learning workflow.

### **Main Features:**
- **DVC** is used to manage dataset versions and large files.
- **MLFlow** is used for logging experiments, tracking models, and versioning.
- **TensorBoard** is used for visualizing training metrics and model performance.

---

## Dependencies

This project requires Python 3.8+ and the following dependencies:

- `dvc` - For data version control and handling large datasets.
- `mlflow` - For experiment tracking and model management.
- `torch` - PyTorch for deep learning tasks.
- `torchvision` - For image transformations and pre-trained models.
- `tensorflow` - For TensorBoard visualization.
- `scikit-learn` - For metrics and model evaluation.
- `pandas` - For data manipulation.
- `yaml` - For loading configuration files.

### To install all dependencies, you can use the provided `requirements.txt`:

1. **Set up a virtual environment** (optional but recommended):
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # For Linux/macOS
    # OR
    myenv\Scripts\activate  # For Windows
    ```

2. **Install dependencies** using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Run the Code

### Step 1: Extract the Dataset (One-Time Setup)

The first step is to download, extract, and organize the **Oxford-IIIT Pet Dataset** into the appropriate structure. Run the following script to perform this:

```bash
python src/data_util.py
```
### Step 2: Run the Pipeline

The next step is to run the DVC pipeline defined in dvc.yaml, it will 
1. preprocess the dataset (split into train/test), 
2. run models with the combination of hyperparameters defined in params.yaml, save them into models/
3. evaluate all the models created and save into metrics/results.json

```bash
dvc repro
```

### Step 3: Visualize

You can check the visualization of tensorboard and mlflow here:

```bash
    tensorboard --logdir=logs/
```
and 
```bash
    mlflow ui
```
