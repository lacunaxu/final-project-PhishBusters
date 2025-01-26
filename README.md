[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/q4BQ8R99)
# DSCI 510 Final Project

## Fraudulent Website Detection and Analysis
Develop a solution to identify and analyze fraudulent websites, leveraging machine learning and data analysis to assist cybersecurity teams and users in safer browsing.

## Team Members (Name and Student IDs)
- **Chenyi Weng**
  - **USC ID**: 3769237784
  - **GitHub Username**: MONA100421
  - **Email**: [wengchen@usc.edu](mailto:wengchen@usc.edu)

- **Zixi Wang**
  - **USC ID**: 2854187591
  - **GitHub Username**: lacunaxu
  - **Email**: [zwang049@usc.edu](mailto:zwang049@usc.edu)

## Instructions to create a virtual environment
- Clone the repository
   - **For macOS/Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **For Windows**:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

## Instructions on how to install the required libraries
Install all dependencies listed in the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Instructions on how to download the data
Run the following script to fetch data from OpenPhish and Tranco. The raw data will be saved in the `data/raw/` directory.
```bash
python src/get_data.py
```

## Instructions on how to clean the data
Run the following script to clean and preprocess the raw data. The processed data will be stored in the `data/processed/` directory.
```bash
python src/clean_data.py
```

## Instrucions on how to run analysis code
Train and evaluate machine learning models using the following script. Ensure that the file `final_test_features.csv` exists in the `data/processed/` directory:

#### ***Skipping Data Generation**
If running `clean_data.py` or `get_data.py` takes too long, it's highly recommended to skip the data generation step:

```bash
cd /path/to/src
python run_analysis.py
```
- How to Switch Back to Full Data Generation & Create Visualizations
  If you decide to fully regenerate the data using `clean_data.py` or `get_data.py`, ensure that you **uncomment** the relevant sections of the code that were commented out for testing purposes. This ensures the scripts generate the necessary data from scratch.

```bash
python src/get_data.py
python src/clean_data.py
python src/run_analysis.py
python src/visualize_results.py
```

## Instructions on how to create visualizations
Create visualizations based on the analysis results. The output will be saved in the `results/visualizations/` directory:
```bash
python visualize_results.py
```

Visualizations include:
1. **Box Plot**: Compare URL length distributions for phishing and legitimate websites.
2. **Scatter Plot**: Explore correlations between features like URL length and risk scores.
3. **Word Cloud**: Highlight frequently used phishing-related terms.
4. **Heatmap**: Depict feature correlations and interactions.
5. **ROC Curve**: Evaluate model effectiveness.
6. **Confusion Matrix**: Summarize model classification performance.

### *Additional Analysis*
For more analysis plots and detailed explanations, refer to the following Jupyter Notebook:

- **Relative Path**: `results/visualizations/visualizations.ipynb`

## **Common Issues and Solutions**
### *1. Helper Utility File & Model Utility File*
Before running the scripts, ensure the following utility files are correctly placed and executable:

Helper Utility File:
- Path: `src/utils/helpers.py`
- This file includes support functions for data preprocessing and feature engineering.

```bash
python src/utils/helpers.py
```
Model Utility File:
 - Path: `src/utils/models.py`
 - This file contains the core functions for machine learning model training and evaluation.

```bash
python src/utils/helpers.py
```

- What it does:
  Trains multiple machine learning models (KNN, Logistic Regression, Random Forest, SVM, etc.).
  - Evaluate model performance, including metrics like accuracy and AUC-ROC.
  - Saves the results to the results/ directory for further analysis.

- Important Notes:
  - Ensure the data/processed/final_test_features.csv file exists before running this script.
  - For optimal performance, verify that all dependencies are correctly installed.

### *2. Path Issues*
If you encounter errors related to file paths:
   - Verify that you are running scripts from the correct directory.
   - Check the current working directory:
     ```bash
     import os
     print(os.getcwd())
     ```
   - If needed, navigate to the `src` directory before running scripts:
     ```bash
     cd /path/to/final-project24-PhishBusters-main/src
     ```

### *3. XGBoost Installation Issues on macOS*
If `clean_data.py` or `run_analysis.py` fails with an error about `libomp`:
   - Install the missing dependency using Homebrew:
     ```bash
     brew install libomp
     ```

## **Tested Environment**
The project has been tested with:
- **Python Version**: 3.12.x
- **Operating System**: macOS (with `libomp` installed via Homebrew)
- **Virtual Environment**: `venv`

If you encounter issues in other environments, ensure all dependencies are correctly installed.

## **Acknowledgements**
Special thanks to USC DSCI 510 for providing the framework and guidelines for this project.
