# Blood-Brain-Barrier-BBB-prediction
This project performs molecular classification using chemical descriptors and machine learning algorithms, including Random Forest, SVM, and XGBoost. It leverages PaDEL-Descriptor for fingerprint generation and evaluates models with metrics like accuracy, F1 score, and more.

## Table of Contents
* Project Overview
* Features
* Installation
* Usage
* Model Evaluation

## Project Overview
This project processes molecular data to predict the permeability of molecules across the blood-brain barrier (BBB) using the PaDEL-Descriptor tool to extract chemical fingerprints. The dataset is used to train machine learning models, and performance metrics such as accuracy, F1 score, and Matthews correlation coefficient are computed to evaluate the models.

## Features
* Load and preprocess molecular data.
* Generate molecular fingerprints using PaDEL-Descriptor.
* Train Random Forest, SVM, and XGBoost models.
* Evaluate models using multiple metrics.
* Feature importance using SHAP values.



## Installation
Prerequisites
* Python 3.x
* pandas, numpy, scikit-learn, padelpy, matplotlib, seaborn, shap

To install the required libraries by running:
```py
pip install pandas numpy scikit-learn padelpy matplotlib seaborn shap
```

## Data and Descriptors
* Load your molecular data in a .tsv file.
* Generate molecular fingerprints using PaDEL-Descriptor.
* The descriptors are saved in CSV format for further analysis.

### Step 1: Load and Process Data
```py
data = pd.read_csv('your_dataset.tsv', sep='\t')
data['BBB+/BBB-'] = data['BBB+/BBB-'].map({'BBB+': 1, 'BBB-': 0})
df = pd.concat([data['SMILES'], data['BBB+/BBB-']], axis=1)
df.to_csv('molecule.smi', sep='\t', index=False, header=False)
```
### Step 2: Generate Molecular Fingerprints
```py
padeldescriptor(mol_dir='molecule.smi',
                d_file='MACCS.csv',
                descriptortypes=fp['MACCS'],
                fingerprints=True)
```
### Step 3: Train Models and Evaluate
```py 
pipeline = make_pipeline(SimpleImputer(strategy='mean'), RandomForestClassifier(random_state=42))
cross_val_score(pipeline, X, y, cv=5, scoring='f1')
```

### Step 4: Feature Selection and SHAP Analysis
* Perform recursive feature elimination.
* Compute SHAP values for feature importance and plot summary.




## Model Evaluation
SVM and XGBoost models are evaluated using:
* Accuracy
* F1 Score (weighted)
* Precision
* Recall
* Matthews Correlation Coefficient (MCC)

Results are saved to text files, and confusion matrices and ROC curves are plotted.

