AI-enhanced CRISPR/Cas12a aptasensor for rapid and ultrasensitive detection of kanamycin in milk
=========================================

.. image:: https://img.shields.io/badge/Python-3.7%2B-blue
   :target: https://www.python.org/
.. image:: https://img.shields.io/badge/Library-XGBoost-green
   :target: https://xgboost.ai/
.. image:: https://img.shields.io/badge/Library-scikit--learn-yellowgreen
   :target: https://scikit-learn.org/

This repository is associated with the paper "AI-enhanced CRISPR/Cas12a aptasensor for rapid and ultrasensitive detection of kanamycin in milk". The repository provides the key code and tools used in the paper, facilitating reproducibility and further research in this area. Advanced XGBoost implementation for time series binary classification with comprehensive performance analysis and visualization capabilities.

Features
--------
- Time series data preprocessing and standardization
- Advanced XGBoost model training with early stopping
- Detailed performance metrics (Accuracy, AUC, F1-score, Precision, Recall)
- 19+ professional visualizations including:
  - ROC curves comparison
  - Precision-Recall curves
  - Feature importance analysis
  - Confusion matrices
  - Probability distributions
  - Performance radar charts
  - Learning curves
  - Feature correlation networks
- Cross-validation with stability analysis
- Comprehensive performance report tables
- Automatic handling of Excel data inputs

Installation
------------
1. Clone the repository::

    git clone https://github.com/yourusername/xgboost-time-series-classification.git
    cd xgboost-time-series-classification

2. Install required packages::

    pip install -r requirements.txt

   Or manually install::

    pip install numpy pandas matplotlib seaborn scikit-learn xgboost openpyxl

Usage
-----
Basic Execution
~~~~~~~~~~~~~~~
Run the main analysis notebook::

    jupyter notebook XGBoost-2(Fig-En).ipynb

Data Preparation
~~~~~~~~~~~~~~~~
Place your time series data in an Excel file named ``DATA-5.xlsx`` with:

- 5 feature columns (X1 to X5)
- 1 label column (binary 0/1)
- Sample data structure:

+----+--------+--------+--------+--------+--------+-------+
| ID |   X1   |   X2   |   X3   |   X4   |   X5   | label |
+====+========+========+========+========+========+=======+
| 1  | 910.23 | 2174.87| 3399.19| 4568.75| 5669.20|   1   |
+----+--------+--------+--------+--------+--------+-------+
| 2  | 840.10 | 2221.61| 3554.34| 4828.21| 6014.50|   1   |
+----+--------+--------+--------+--------+--------+-------+

Code Execution
~~~~~~~~~~~~~~
The analysis will automatically:

1. Load and preprocess data
2. Split into train/validation/test sets
3. Train XGBoost model
4. Generate performance metrics
5. Create visualizations in ``/xgboost_plots`` directory

Outputs
-------
Performance Metrics
~~~~~~~~~~~~~~~~~~~
::

    === Final Test Results ===
    Test Accuracy: 1.0000
    Test AUC: 1.0000
    Test F1-Score: 1.0000
    Cross-validation AUC Mean: 0.9970 (±0.0060)

Generated Visualizations
~~~~~~~~~~~~~~~~~~~~~~~~
+---------------------------------+--------------------------------------------+
| Filename                        | Description                                |
+=================================+============================================+
| ``02_feature_importance.svg``   | Feature importance ranking                 |
+---------------------------------+--------------------------------------------+
| ``03_roc_curves.svg``           | ROC curves comparison                     |
+---------------------------------+--------------------------------------------+
| ``04_precision_recall_curves.svg``| Precision-Recall curves                   |
+---------------------------------+--------------------------------------------+
| ``05_confusion_matrix.svg``     | Confusion matrix                          |
+---------------------------------+--------------------------------------------+
| ``06_probability_distribution.svg``| Prediction probability distribution        |
+---------------------------------+--------------------------------------------+
| ``07_performance_radar.svg``    | Performance metrics radar                 |
+---------------------------------+--------------------------------------------+
| ``08_learning_curve.svg``       | Learning curve                            |
+---------------------------------+--------------------------------------------+
| ``09_feature_correlation.svg``  | Feature correlation heatmap               |
+---------------------------------+--------------------------------------------+
| ``15_feature_importance_ranking.svg``| Feature importance ranking              |
+---------------------------------+--------------------------------------------+
| ``17_correlation_network.svg``  | Feature correlation network               |
+---------------------------------+--------------------------------------------+
| ``19_performance_report_table.svg``| Detailed performance report              |
+---------------------------------+--------------------------------------------+

Sample Output
-------------
.. image:: https://github.com/MaC14a1-2025/Tang2025/blob/main/xgboost-output/02_feature_importance.svg
   :width: 45%
   :alt: feature_importance

.. image:: https://github.com/MaC14a1-2025/Tang2025/blob/main/xgboost-output/08_learning_curve.svg
   :width: 45%
   :alt: learning_curve

.. image:: https://github.com/MaC14a1-2025/Tang2025/blob/main/xgboost-output/19_performance_report_table.svg
   :width: 45%
   :alt: performance_report_table



Customization
-------------
Modify these parameters in the notebook for customization::

    # Model parameters
    self.model = xgb.XGBClassifier(
        n_estimators=200,       # Number of trees
        max_depth=6,            # Tree depth
        learning_rate=0.1,      # Learning rate
        subsample=0.8,          # Subsample ratio
        colsample_bytree=0.8,   # Feature subsample ratio
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Data splitting ratios
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4     # 40% for validation+test
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5  # Split temp set equally
    )

Requirements
------------
- Python 3.7+
- numpy >= 1.19.5
- pandas >= 1.2.0
- matplotlib >= 3.3.4
- seaborn >= 0.11.1
- scikit-learn >= 0.24.1
- xgboost >= 1.4.0
- openpyxl >= 3.0.7

License
-------
MIT License. See ``LICENSE`` file for details.
