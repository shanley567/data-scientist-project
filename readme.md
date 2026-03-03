# Machine Learning Pipeline

This project is intended to create a reusable ML modelling pipeline for structured table data, including data loading, preprocessing, model training, hyperparamerer tuning, evaluation, validation and persistance for deployment.

The project is build in a modular layour for reusability.

```plaintext
Folder PATH listing for volume OS
Volume serial number is 22AD-7B27
C:.
|   .gitignore
|   main.py
|   readme.md
|   requirements.txt
|   
+---data
|   +---processed
|   \---raw
|           abalone.csv
|           candy.csv
|           cereal.csv
|           concrete.csv
|           diamonds.csv
|           forestfires.csv
|           fuel.csv
|           hotel.csv
|           housing.csv
|           ion.csv
|           red-wine.csv
|           songs.csv
|           spotify.csv
|   
+---models
|       concrete_strength_model.joblib
|   
+---notebooks
|       data_exploration.ipynb
|   
\---src
    |   data_loader.py
    |   evaluate.py
    |   models.py
    |   preprocessing.py
    |   train.py
    |   utils.py
    |   
    \---__pycache__
            data_loader.cpython-314.pyc
            evaluate.cpython-314.pyc
            models.cpython-314.pyc
            preprocessing.cpython-314.pyc
            train.cpython-314.pyc
            utils.cpython-314.pyc
```

Each components responsability:

### SRC

* **data_loader.py** — loads and cleans the dataset, separates the target.
* **preprocessing.py** — builds the preprocessing pipeline (imputers, encoders, scalers).
* **models.py** — defines model constructors and hyperparameter grids.
* **train.py** — trains models using GridSearchCV and saves the best estimator.
* **evaluate.py** — evaluates a trained or saved model on validation data.
* **utils.py** — helper functions for saving/loading models and general utilities.
* **main.py** — orchestrates the full workflow end‑to‑end

### Datasets

A collection of tabulated data in CSV files for model testing.

### Preprocessing

Preprocessing is planted in a scikit-learn pipeline to prevent data leakage and consistant transformations.

Specific data columns (features) are selected:

- Numerical (float64 & int)

  - Uses median imputation to fill missing values
  - Standard scaling (sets mean=0, and standard deviation=1), this is mainly for neural networks but does not hinder regressions
- Categorical

  - Most Frequent imputation for missing values
  - One Hot encoding for features with cardinality<10. Also ignores unknown featured in prediction that were not in training sets.

  ### Model Training
- Currently consists of random forrest regressor as the standard model but may change in later versions.
- GridSearchCV is used for hyperparameter tuning of various models to select the best model
- Currently used cross validation "`CV`" in the grid
- Mean Absolute Error (MAE) is the current scoring metric

The best model is saved to `/models` directory.

### Evaluation

Evaluation module computes:

- Validation MAE
- Predictions on unseen data
- Optionally loads model for inference

Example:

```python
from src.evaluate import evaluate
mae = evaluate(model=best_model, X_valid=X_valid, y_valid=y_valid)
```

### Running the Project

1. Specify the dataset path and target column in the main.py file, in the load_data function
2. Run the main script
   1. Loads and preprocesses data
   2. Trains a tuned model
   3. Saved best model to models/ directory
   4. Evaluated the performance

### Loading Saved Models

Using the utility.py functions

```python
from src.utils import load_model

model = load_model() # Specify model
preds = model.predict(new_data) # Speficy file path
```

### Project Extensions

This modelar project is designed to be reusable on different datasets and can add:

- New model types (XGBRegressions, NN...)
- Feature Engineering Steps
- Model type Comparison
- Deployments
