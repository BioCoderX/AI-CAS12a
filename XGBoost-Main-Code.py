import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, f1_score, precision_score, recall_score)
from matplotlib.colors import TwoSlopeNorm
import warnings
import os

warnings.filterwarnings('ignore')

# XGBoost import
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
    print("✓ XGBoost available, version:", xgb.__version__)
except ImportError:
    XGBOOST_AVAILABLE = False
    print("✗ XGBoost not installed, please run: pip install xgboost")
    exit()

# Set English font support
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)


# Generate mock dataset (375 samples, 5 time features)
def generate_sample_data():
    """Generate simulated time series data"""
    n_samples = 375

    # Generate basic time features
    X1 = np.random.normal(10, 2, n_samples)  # 1-minute feature
    X2 = X1 + np.random.normal(0, 1, n_samples)  # 2-minute feature (correlated with X1)
    X3 = X2 + np.random.normal(0, 1.5, n_samples)  # 3-minute feature
    X4 = X3 + np.random.normal(0, 1, n_samples)  # 4-minute feature
    X5 = X4 + np.random.normal(0, 2, n_samples)  # 5-minute feature

    # Create feature matrix
    X = np.column_stack([X1, X2, X3, X4, X5])

    # Generate labels (based on feature combination)
    trend = (X5 - X1) / 4  # Calculate trend
    mean_value = np.mean(X, axis=1)  # Mean value

    # Generate labels based on trend and mean value
    prob_positive = 1 / (1 + np.exp(-(0.3 * trend + 0.1 * mean_value - 2)))
    y = np.random.binomial(1, prob_positive, n_samples)

    return X, y


def load_excel_data(file_path='DATA-5.xlsx', sheet_name=0):
    """Load data from Excel file"""
    try:
        # Try different engines to read Excel file
        engines = ['openpyxl', 'xlrd']
        df = None

        for engine in engines:
            try:
                print(f"Trying to read file using {engine} engine...")
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine=engine)
                print(f"✓ Successfully read data using {engine}")
                break
            except Exception as e:
                print(f"✗ {engine} engine failed: {e}")
                continue

        if df is None:
            raise Exception("All engines failed to read the file")

        print(f"Successfully read data, shape: {df.shape}")
        print(f"Column names: {list(df.columns)}")

        # Display first few rows
        print("\nData preview:")
        print(df.head())

        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            print("\nMissing values statistics:")
            print(missing_values[missing_values > 0])
            print("Removing rows with missing values...")
            df = df.dropna()
            print(f"Shape after removal: {df.shape}")

        # Automatically identify feature columns and label column
        feature_cols = []
        label_col = None

        # Look for X1-X5 feature columns (case insensitive)
        for col in df.columns:
            col_str = str(col).strip().upper()
            if col_str in ['X1', 'X2', 'X3', 'X4', 'X5']:
                feature_cols.append(col)

        # Sort by X1,X2,X3,X4,X5 order
        feature_order = {'X1': 1, 'X2': 2, 'X3': 3, 'X4': 4, 'X5': 5}
        feature_cols.sort(key=lambda x: feature_order.get(str(x).strip().upper(), 999))

        # Look for label column
        for col in df.columns:
            col_str = str(col).strip().lower()
            if col_str in ['label', 'y', 'target', 'class', 'lable']:
                label_col = col
                break

        # If standard column names not found, smart guess
        if len(feature_cols) != 5:
            print(f"\nOnly found {len(feature_cols)} standard feature columns: {feature_cols}")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 5:
                feature_cols = numeric_cols[:5]
                print(f"Auto-selecting first 5 numeric columns as features: {feature_cols}")
            else:
                return None, None

        if label_col is None:
            remaining_cols = [col for col in df.columns if col not in feature_cols]
            if remaining_cols:
                label_col = remaining_cols[0]
                print(f"Auto-selecting column as label: {label_col}")
            else:
                return None, None

        # Extract features and labels
        X = df[feature_cols].values
        y = df[label_col].values

        # Process labels
        unique_labels = np.unique(y)
        if set(unique_labels) == {0, 1} or set(unique_labels) == {0.0, 1.0}:
            y = y.astype(int)
        elif len(unique_labels) == 2:
            min_val, max_val = min(unique_labels), max(unique_labels)
            y = (y == max_val).astype(int)
        else:
            return None, None

        print(f"\nFinal data dimensions: X={X.shape}, y={y.shape}")
        print(
            f"Label distribution: Positive={np.sum(y)} ({100 * np.mean(y):.1f}%), Negative={len(y) - np.sum(y)} ({100 * (1 - np.mean(y)):.1f}%)")

        return X, y

    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return None, None
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None, None


def prepare_data(X, y):
    """Data standardization and split into train/validation/test sets"""
    # Ensure input is numpy array format
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values

    X = np.array(X)
    y = np.array(y)

    # First split into train and temp sets (test + validation)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    # Then split temp set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set size: {X_train_scaled.shape[0]}")
    print(f"Validation set size: {X_val_scaled.shape[0]}")
    print(f"Test set size: {X_test_scaled.shape[0]}")
    print(f"Training set positive ratio: {np.mean(y_train):.3f}")
    print(f"Validation set positive ratio: {np.mean(y_val):.3f}")
    print(f"Test set positive ratio: {np.mean(y_test):.3f}")

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test, scaler)


class XGBoostDetailedModel:
    """Enhanced XGBoost model with detailed training process monitoring and visualization"""

    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        self.name = "XGBoost Detailed Analysis Model"
        self.feature_importance = None
        self.training_history = {}
        self.feature_names = ['X1 (1-minute)', 'X2 (2-minute)', 'X3 (3-minute)', 'X4 (4-minute)', 'X5 (5-minute)']

    def train(self, X_train, y_train, X_val, y_val):
        """Train model and record detailed process"""
        print(f"Starting training {self.name}...")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Create validation set
        eval_set = [(X_train, y_train), (X_val, y_val)]
        eval_names = ['train', 'validation']

        try:
            # Train with early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=20,
                verbose=10
            )
            print("✓ Training completed")

            # Get training history
            self.training_history = self.model.evals_result()

        except Exception as e:
            print(f"Training error: {e}")
            # Use basic training method
            self.model.fit(X_train, y_train)
            print("✓ Completed using basic training method")

        # Validation evaluation
        val_pred = self.model.predict(X_val)
        val_prob = self.model.predict_proba(X_val)[:, 1]
        val_accuracy = accuracy_score(y_val, val_pred)
        val_auc = roc_auc_score(y_val, val_prob)
        val_f1 = f1_score(y_val, val_pred)

        print(f"\nValidation performance:")
        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"AUC: {val_auc:.4f}")
        print(f"F1-score: {val_f1:.4f}")

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
            importance_dict = dict(zip(self.feature_names, self.feature_importance))
            print(f"\nFeature importance:")
            for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
                print(f"{feature}: {importance:.4f}")

        return val_accuracy, val_auc, val_f1

    def predict(self, X):
        """Predict"""
        pred = self.model.predict(X)
        prob = self.model.predict_proba(X)[:, 1]
        return pred, prob

    def cross_validate(self, X, y, cv=5):
        """Cross validation"""
        print(f"\nPerforming {cv}-fold cross validation...")
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='roc_auc')
        print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        return cv_scores


def save_individual_plots(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Generate and save individual plots as SVG files"""

    # Create output directory if it doesn't exist
    import os
    output_dir = "xgboost_plots"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving individual plots to {output_dir}/ directory...")

    # Predict on all datasets
    train_pred, train_prob = model.predict(X_train)
    val_pred, val_prob = model.predict(X_val)
    test_pred, test_prob = model.predict(X_test)

    # 1. Training loss curve
    if model.training_history:
        fig, ax = plt.subplots(figsize=(10, 6))
        train_loss = model.training_history.get('train', {}).get('logloss', [])
        val_loss = model.training_history.get('validation', {}).get('logloss', [])

        if train_loss and val_loss:
            epochs = range(1, len(train_loss) + 1)
            ax.plot(epochs, train_loss, 'b-', label='Training loss', linewidth=2)
            ax.plot(epochs, val_loss, 'r-', label='Validation loss', linewidth=2)
            ax.set_xlabel('Training rounds')
            ax.set_ylabel('Logarithmic loss')
            ax.set_title('Loss Curve During Training', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Mark best iteration
            if hasattr(model.model, 'best_iteration'):
                best_iter = model.model.best_iteration
                ax.axvline(x=best_iter, color='green', linestyle='--', alpha=0.7,
                           label=f'Best round: {best_iter}')
                ax.legend()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/01_training_loss_curve.svg', format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 01_training_loss_curve.svg")

    # 2. Feature importance plot
    if model.feature_importance is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        importance_df = pd.DataFrame({
            'feature': model.feature_names,
            'importance': model.feature_importance
        }).sort_values('importance', ascending=True)

        bars = ax.barh(importance_df['feature'], importance_df['importance'],
                       color=plt.cm.viridis(np.linspace(0, 1, len(importance_df))))
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance Ranking', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, importance_df['importance'])):
            ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{value:.3f}', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/02_feature_importance.svg', format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 02_feature_importance.svg")

    # 3. ROC curves comparison
    fig, ax = plt.subplots(figsize=(10, 8))

    # Training set ROC
    fpr_train, tpr_train, _ = roc_curve(y_train, train_prob)
    auc_train = roc_auc_score(y_train, train_prob)

    # Validation set ROC
    fpr_val, tpr_val, _ = roc_curve(y_val, val_prob)
    auc_val = roc_auc_score(y_val, val_prob)

    # Test set ROC
    fpr_test, tpr_test, _ = roc_curve(y_test, test_prob)
    auc_test = roc_auc_score(y_test, test_prob)

    ax.plot(fpr_train, tpr_train, 'b-', linewidth=2, label=f'Training (AUC = {auc_train:.3f})')
    ax.plot(fpr_val, tpr_val, 'g-', linewidth=2, label=f'Validation (AUC = {auc_val:.3f})')
    ax.plot(fpr_test, tpr_test, 'r-', linewidth=2, label=f'Test (AUC = {auc_test:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_roc_curves.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 03_roc_curves.svg")

    # 4. Precision-Recall curves
    fig, ax = plt.subplots(figsize=(10, 8))

    precision_train, recall_train, _ = precision_recall_curve(y_train, train_prob)
    ap_train = average_precision_score(y_train, train_prob)

    precision_val, recall_val, _ = precision_recall_curve(y_val, val_prob)
    ap_val = average_precision_score(y_val, val_prob)

    precision_test, recall_test, _ = precision_recall_curve(y_test, test_prob)
    ap_test = average_precision_score(y_test, test_prob)

    ax.plot(recall_train, precision_train, 'b-', linewidth=2, label=f'Training (AP = {ap_train:.3f})')
    ax.plot(recall_val, precision_val, 'g-', linewidth=2, label=f'Validation (AP = {ap_val:.3f})')
    ax.plot(recall_test, precision_test, 'r-', linewidth=2, label=f'Test (AP = {ap_test:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_precision_recall_curves.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 04_precision_recall_curves.svg")

    # 5. Confusion matrix
    plt.rcParams['font.size'] = 18  # 全局默认字体大小设置为10
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"fontsize": 28},
                xticklabels=['0', '1'], yticklabels=['0', '1'])
    ax.set_xlabel('Predicted Label', fontsize=28)
    ax.set_ylabel('True Label', fontsize=28)
    ax.set_title('Test Set Confusion Matrix', fontsize=28, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_confusion_matrix.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 05_confusion_matrix.svg")

    # 6. Prediction probability distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    pos_probs = test_prob[y_test == 1]
    neg_probs = test_prob[y_test == 0]

    ax.hist(neg_probs, bins=30, alpha=0.7, label='Negative', color='red', density=True)
    ax.hist(pos_probs, bins=30, alpha=0.7, label='Positive', color='blue', density=True)
    ax.set_xlabel('Prediction Probability')
    ax.set_ylabel('Density')
    ax.set_title('Predicted Probability Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_probability_distribution.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 06_probability_distribution.svg")

    # 7. Performance metrics radar chart
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    # Calculate various performance metrics
    metrics_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1-score', 'Specificity']

    accuracy_test = accuracy_score(y_test, test_pred)
    precision_test = precision_score(y_test, test_pred)
    recall_test = recall_score(y_test, test_pred)
    f1_test = f1_score(y_test, test_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
    specificity_test = tn / (tn + fp)

    metrics_values = [accuracy_test, auc_test, precision_test, recall_test, f1_test, specificity_test]

    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    angles += angles[:1]  # Close the plot
    metrics_values += metrics_values[:1]  # Close the plot

    ax.plot(angles, metrics_values, 'o-', linewidth=2, color='red')
    ax.fill(angles, metrics_values, alpha=0.25, color='red')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 1)
    ax.set_title('Test Set Performance Metrics Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_performance_radar.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 07_performance_radar.svg")

    # 8. Learning curve
    fig, ax = plt.subplots(figsize=(10, 6))
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    val_scores = []

    for size in train_sizes:
        n_samples = int(size * len(X_train))
        if n_samples < 10:
            continue

        # Subsample training
        indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_sub = X_train[indices]
        y_sub = y_train[indices]

        # Train temporary model
        temp_model = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            random_state=42, eval_metric='logloss', use_label_encoder=False
        )
        temp_model.fit(X_sub, y_sub)

        # Evaluate
        train_pred_temp = temp_model.predict_proba(X_sub)[:, 1]
        val_pred_temp = temp_model.predict_proba(X_val)[:, 1]

        train_scores.append(roc_auc_score(y_sub, train_pred_temp))
        val_scores.append(roc_auc_score(y_val, val_pred_temp))

    if train_scores and val_scores:
        valid_sizes = train_sizes[:len(train_scores)]
        ax.plot(valid_sizes * len(X_train), train_scores, 'b-', linewidth=4, label='Training AUC')
        ax.plot(valid_sizes * len(X_train), val_scores, 'r-', linewidth=4, label='Validation AUC')
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.set_xlabel('Training Sample Size', fontsize=28)
        ax.set_ylabel('AUC Score', fontsize=28)
        ax.set_title('Learning Curve', fontsize=28, fontweight='bold')
        ax.legend(fontsize=24)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_learning_curve.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 08_learning_curve.svg")

    # 9. Feature correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    feature_data = pd.DataFrame(X_train, columns=[f'X{i + 1}' for i in range(X_train.shape[1])])
    corr_matrix = feature_data.corr()
    sns.heatmap(corr_matrix, annot=True,
                cmap=sns.color_palette("vlag", as_cmap=True),
                norm=TwoSlopeNorm(vmin=0.9, vcenter=0.95, vmax=1),
                center=0, ax=ax,
                square=True, fmt='.2f',
                linewidths=0.6,
                linecolor='white',
                annot_kws={'fontsize': 10, 'fontweight': 'bold'},
                cbar_kws={'shrink': 0.8, 'ticks': [-1, -0.5, 0, 0.5, 1]})
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/09_feature_correlation.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 09_feature_correlation.svg")

    # 10. First decision tree visualization or feature importance pie chart
    fig, ax = plt.subplots(figsize=(12, 8))
    try:
        # Try to plot the first decision tree
        xgb.plot_tree(model.model, num_trees=0, ax=ax, rankdir='TB')
        ax.set_title('XGBoost First Decision Tree Structure', fontsize=14, fontweight='bold')
    except:
        # If tree plotting fails, show feature importance pie chart
        if model.feature_importance is not None:
            ax.pie(model.feature_importance, labels=model.feature_names, autopct='%1.1f%%',
                   startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(model.feature_names))))
            ax.set_title('Feature Importance Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/10_tree_or_importance_pie.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 10_tree_or_importance_pie.svg")

    return {
        'train_auc': auc_train,
        'val_auc': auc_val,
        'test_auc': auc_test,
        'test_accuracy': accuracy_test,
        'test_f1': f1_test,
        'test_precision': precision_test,
        'test_recall': recall_test
    }


def plot_training_progress(model):
    """Plot detailed training progress"""
    if not model.training_history:
        print("No training history data available for visualization")
        return

    output_dir = "xgboost_plots"

    train_loss = model.training_history.get('train', {}).get('logloss', [])
    val_loss = model.training_history.get('validation', {}).get('logloss', [])

    if not train_loss or not val_loss:
        print("Training history data incomplete")
        return

    epochs = range(1, len(train_loss) + 1)

    # Training loss curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    ax.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss')
    ax.set_xlabel('Training Rounds')
    ax.set_ylabel('Logarithmic Loss')
    ax.set_title('Training Process Loss Curve', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark best iteration
    if hasattr(model.model, 'best_iteration') and model.model.best_iteration:
        best_iter = model.model.best_iteration
        ax.axvline(x=best_iter, color='green', linestyle='--', alpha=0.7)
        ax.text(best_iter, min(val_loss), f'Best Round: {best_iter}',
                rotation=90, verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/11_detailed_training_loss.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 11_detailed_training_loss.svg")

    # Loss improvement trend
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs[1:], np.diff(train_loss), 'b-', label='Training Loss Improvement', alpha=0.7)
    ax.plot(epochs[1:], np.diff(val_loss), 'r-', label='Validation Loss Improvement', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Training Rounds')
    ax.set_ylabel('Loss Improvement')
    ax.set_title('Loss Improvement Trend', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/12_loss_improvement_trend.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 12_loss_improvement_trend.svg")

    # Overfitting detection
    fig, ax = plt.subplots(figsize=(10, 6))
    overfitting = np.array(val_loss) - np.array(train_loss)
    ax.plot(epochs, overfitting, 'purple', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.fill_between(epochs, overfitting, 0, alpha=0.3, color='purple')
    ax.set_xlabel('Training Rounds')
    ax.set_ylabel('Validation Loss - Training Loss')
    ax.set_title('Overfitting Detection (Higher Gap = More Overfitting)', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/13_overfitting_detection.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 13_overfitting_detection.svg")

    # Training stability
    fig, ax = plt.subplots(figsize=(10, 6))
    window_size = min(10, len(train_loss) // 4)
    if window_size > 1:
        train_smooth = pd.Series(train_loss).rolling(window=window_size).std()
        val_smooth = pd.Series(val_loss).rolling(window=window_size).std()

        ax.plot(epochs, train_smooth, 'b-', label='Training Loss Volatility', alpha=0.7)
        ax.plot(epochs, val_smooth, 'r-', label='Validation Loss Volatility', alpha=0.7)
        ax.set_xlabel('Training Rounds')
        ax.set_ylabel('Loss Standard Deviation (Rolling Window)')
        ax.set_title('Training Stability Analysis', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/14_training_stability.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 14_training_stability.svg")


def analyze_feature_interactions(model, X, feature_names):
    """Analyze feature interactions"""
    if model.feature_importance is None:
        print("Model feature importance data not available")
        return

    output_dir = "xgboost_plots"

    # Feature importance comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    importance = model.feature_importance
    indices = np.argsort(importance)[::-1]

    ax.bar(range(len(importance)), importance[indices],
           color=plt.cm.viridis(np.linspace(0, 1, len(importance))))
    ax.set_title('Feature Importance Ranking', fontweight='bold', fontsize=14)
    ax.set_xlabel('Feature Rank')
    ax.set_ylabel('Importance Score')
    ax.set_xticks(range(len(importance)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/15_feature_importance_ranking.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 15_feature_importance_ranking.svg")

    # Feature distribution comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot([X[:, i] for i in range(X.shape[1])],
               labels=[f'X{i + 1}' for i in range(X.shape[1])])
    ax.set_title('Feature Value Distribution', fontweight='bold', fontsize=14)
    ax.set_ylabel('Feature Values')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/16_feature_distribution.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 16_feature_distribution.svg")

    # Feature correlation network graph
    fig, ax = plt.subplots(figsize=(10, 10))
    feature_data = pd.DataFrame(X, columns=[f'X{i + 1}' for i in range(X.shape[1])])
    corr_matrix = feature_data.corr()

    # Create network graph layout
    G_pos = {}
    angle_step = 2 * np.pi / len(feature_names)
    for i, name in enumerate(feature_names):
        angle = i * angle_step
        G_pos[name] = (np.cos(angle), np.sin(angle))

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    # Draw nodes
    for name, (x, y) in G_pos.items():
        ax.scatter(x, y, s=1000, c='lightblue', alpha=0.7)
        ax.text(x, y, name.split('(')[0], ha='center', va='center', fontweight='bold')

    # Draw connection lines (correlations)
    for i, name1 in enumerate(feature_names):
        for j, name2 in enumerate(feature_names):
            if i < j:  # Avoid duplicate drawing
                corr = abs(corr_matrix.iloc[i, j])
                if corr > 0.3:  # Only show strong correlations
                    x1, y1 = G_pos[name1]
                    x2, y2 = G_pos[name2]
                    ax.plot([x1, x2], [y1, y2], 'r-', alpha=corr, linewidth=corr * 3)

    ax.set_title('Feature Correlation Network (Line thickness shows correlation strength)',
                 fontweight='bold', fontsize=14)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/17_correlation_network.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 17_correlation_network.svg")

    # Cumulative importance contribution
    fig, ax = plt.subplots(figsize=(10, 6))
    cumsum_importance = np.cumsum(sorted(importance, reverse=True))
    cumsum_importance = cumsum_importance / cumsum_importance[-1] * 100

    ax.plot(range(1, len(cumsum_importance) + 1), cumsum_importance, 'bo-', linewidth=4)
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% Contribution Line', linewidth=4)
    ax.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% Contribution Line', linewidth=4)
    ax.set_xlabel('Number of Features', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_ylabel('Contribution (%)', fontsize=28)  ##Cumulative Importance Contribution
    ax.set_title('Feature Importance Cumulative Contribution', fontweight='bold', fontsize=28)
    ax.legend(fontsize=24)
    ax.grid(True, alpha=0.3)

    # Add value labels
    for i, y in enumerate(cumsum_importance):
        if i == 0 or i == len(cumsum_importance) - 1 or abs(y - 80) < 5 or abs(y - 95) < 5:
            ax.annotate(f'{y:.1f}%', (i + 1, y), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/18_cumulative_importance.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 18_cumulative_importance.svg")


def create_performance_report_table(y_test, test_pred, test_prob):
    """Create detailed performance report table"""
    output_dir = "xgboost_plots"

    # Calculate all metrics
    accuracy_test = accuracy_score(y_test, test_pred)
    precision_test = precision_score(y_test, test_pred)
    recall_test = recall_score(y_test, test_pred)
    f1_test = f1_score(y_test, test_pred)
    auc_test = roc_auc_score(y_test, test_prob)
    ap_test = average_precision_score(y_test, test_prob)

    # Create performance report table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Create detailed performance report
    report = classification_report(y_test, test_pred, output_dict=True)

    report_data = [
        ['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
        ['Negative (0)', f"{report['0']['precision']:.3f}", f"{report['0']['recall']:.3f}",
         f"{report['0']['f1-score']:.3f}", f"{report['0']['support']:.0f}"],
        ['Positive (1)', f"{report['1']['precision']:.3f}", f"{report['1']['recall']:.3f}",
         f"{report['1']['f1-score']:.3f}", f"{report['1']['support']:.0f}"],
        ['Macro Average', f"{report['macro avg']['precision']:.3f}", f"{report['macro avg']['recall']:.3f}",
         f"{report['macro avg']['f1-score']:.3f}", f"{report['macro avg']['support']:.0f}"],
        ['Weighted Average', f"{report['weighted avg']['precision']:.3f}", f"{report['weighted avg']['recall']:.3f}",
         f"{report['weighted avg']['f1-score']:.3f}", f"{report['weighted avg']['support']:.0f}"],
        ['', '', '', '', ''],
        ['Overall Accuracy', f"{accuracy_test:.3f}", '', '', f"{len(y_test)}"],
        ['AUC Score', f"{auc_test:.3f}", '', '', ''],
        ['Average Precision', f"{ap_test:.3f}", '', '', '']
    ]

    table = ax.table(cellText=report_data[1:], colLabels=report_data[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0)

    # Set table style
    for i in range(len(report_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight performance rows
    for i in range(1, 4):  # Negative, Positive, Macro average rows
        for j in range(len(report_data[0])):
            if i <= 2:  # Class rows
                table[(i, j)].set_facecolor('#E8F5E8')
            elif i == 3:  # Macro average row
                table[(i, j)].set_facecolor('#FFF3CD')

    ax.set_title('XGBoost Model Detailed Performance Report', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/19_performance_report_table.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 19_performance_report_table.svg")


def validate_real_data(model, scaler, real_data_path='Real-Data.xlsx'):
    """Validate Real-Data.xlsx using trained model"""
    print("\n=== Validating Real-Data.xlsx ===")

    # Load real data
    X_real, y_real = load_excel_data(real_data_path)
    if X_real is None or y_real is None:
        print("Error loading Real-Data.xlsx. Using simulated data instead.")
        X_real, y_real = generate_sample_data()

    print(f"Real-Data dimensions: X={X_real.shape}, y={y_real.shape}")
    print(
        f"Label distribution: Positive={np.sum(y_real)} ({100 * np.mean(y_real):.1f}%), Negative={len(y_real) - np.sum(y_real)} ({100 * (1 - np.mean(y_real)):.1f}%)")

    # Standardize using the same scaler from training
    X_real_scaled = scaler.transform(X_real)

    # Predict
    real_pred, real_prob = model.predict(X_real_scaled)

    # Calculate metrics
    real_acc = accuracy_score(y_real, real_pred)
    real_auc = roc_auc_score(y_real, real_prob)
    real_f1 = f1_score(y_real, real_pred)
    real_precision = precision_score(y_real, real_pred)
    real_recall = recall_score(y_real, real_pred)

    # Print results
    print("\n=== Real-Data Evaluation Results ===")
    print(f"Accuracy: {real_acc:.4f}")
    print(f"AUC: {real_auc:.4f}")
    print(f"F1-score: {real_f1:.4f}")
    print(f"Precision: {real_precision:.4f}")
    print(f"Recall: {real_recall:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_real, real_pred, target_names=['Negative', 'Positive']))

    # Create output directory
    output_dir = "real_data_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Generate comprehensive visualizations
    generate_real_data_visualizations(model, X_real_scaled, y_real, real_pred, real_prob, output_dir)

    return {
        'accuracy': real_acc,
        'auc': real_auc,
        'f1': real_f1,
        'precision': real_precision,
        'recall': real_recall
    }


def generate_real_data_visualizations(model, X_real, y_real, real_pred, real_prob, output_dir):
    """Generate comprehensive visualizations for real data"""
    print(f"\nGenerating Real-Data visualizations in {output_dir}/ directory...")

    # 1. ROC curve
    fig, ax = plt.subplots(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_real, real_prob)
    auc_score = roc_auc_score(y_real, real_prob)

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Real-Data (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('ROC Curve for Real-Data', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/real_data_roc_curve.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: real_data_roc_curve.svg")

    # 2. Precision-Recall curve
    fig, ax = plt.subplots(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_real, real_prob)
    ap_score = average_precision_score(y_real, real_prob)

    ax.plot(recall, precision, 'r-', linewidth=2, label=f'Real-Data (AP = {ap_score:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve for Real-Data', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/real_data_pr_curve.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: real_data_pr_curve.svg")

    # 3. Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_real, real_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Real-Data Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/real_data_confusion_matrix.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: real_data_confusion_matrix.svg")

    # 4. Prediction probability distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    pos_probs = real_prob[y_real == 1]
    neg_probs = real_prob[y_real == 0]

    ax.hist(neg_probs, bins=30, alpha=0.7, label='Negative', color='red', density=True)
    ax.hist(pos_probs, bins=30, alpha=0.7, label='Positive', color='blue', density=True)
    ax.set_xlabel('Prediction Probability')
    ax.set_ylabel('Density')
    ax.set_title('Real-Data Prediction Probability Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/real_data_probability_distribution.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: real_data_probability_distribution.svg")

    # 5. Performance metrics radar chart
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    # Calculate various performance metrics
    metrics_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1-score', 'Specificity']

    accuracy = accuracy_score(y_real, real_pred)
    precision = precision_score(y_real, real_pred)
    recall = recall_score(y_real, real_pred)
    f1 = f1_score(y_real, real_pred)
    tn, fp, fn, tp = confusion_matrix(y_real, real_pred).ravel()
    specificity = tn / (tn + fp)
    auc = roc_auc_score(y_real, real_prob)

    metrics_values = [accuracy, auc, precision, recall, f1, specificity]

    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    angles += angles[:1]  # Close the plot
    metrics_values += metrics_values[:1]  # Close the plot

    ax.plot(angles, metrics_values, 'o-', linewidth=2, color='purple')
    ax.fill(angles, metrics_values, alpha=0.25, color='purple')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 1)
    ax.set_title('Real-Data Performance Metrics Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/real_data_performance_radar.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: real_data_performance_radar.svg")

    # 6. Feature importance plot (from trained model)
    if model.feature_importance is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        importance_df = pd.DataFrame({
            'feature': model.feature_names,
            'importance': model.feature_importance
        }).sort_values('importance', ascending=True)

        bars = ax.barh(importance_df['feature'], importance_df['importance'],
                       color=plt.cm.viridis(np.linspace(0, 1, len(importance_df))))
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance Ranking', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, importance_df['importance'])):
            ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{value:.3f}', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/real_data_feature_importance.svg', format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: real_data_feature_importance.svg")

    # 7. Real data feature distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    feature_data = pd.DataFrame(X_real, columns=[f'X{i + 1}' for i in range(X_real.shape[1])])

    for i, col in enumerate(feature_data.columns):
        sns.kdeplot(feature_data[col], fill=True, alpha=0.5, label=f'X{i + 1}')

    ax.set_title('Real-Data Feature Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Values')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/real_data_feature_distribution.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: real_data_feature_distribution.svg")

    # 8. Classification report table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Create detailed performance report
    report = classification_report(y_real, real_pred, output_dict=True)

    report_data = [
        ['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
        ['Negative (0)', f"{report['0']['precision']:.3f}", f"{report['0']['recall']:.3f}",
         f"{report['0']['f1-score']:.3f}", f"{report['0']['support']:.0f}"],
        ['Positive (1)', f"{report['1']['precision']:.3f}", f"{report['1']['recall']:.3f}",
         f"{report['1']['f1-score']:.3f}", f"{report['1']['support']:.0f}"],
        ['Macro Average', f"{report['macro avg']['precision']:.3f}", f"{report['macro avg']['recall']:.3f}",
         f"{report['macro avg']['f1-score']:.3f}", f"{report['macro avg']['support']:.0f}"],
        ['Weighted Average', f"{report['weighted avg']['precision']:.3f}", f"{report['weighted avg']['recall']:.3f}",
         f"{report['weighted avg']['f1-score']:.3f}", f"{report['weighted avg']['support']:.0f}"],
        ['', '', '', '', ''],
        ['Overall Accuracy', f"{accuracy:.3f}", '', '', f"{len(y_real)}"],
        ['AUC Score', f"{auc:.3f}", '', '', ''],
        ['Average Precision', f"{ap_score:.3f}", '', '', '']
    ]

    table = ax.table(cellText=report_data[1:], colLabels=report_data[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0)

    # Set table style
    for i in range(len(report_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight performance rows
    for i in range(1, 4):  # Negative, Positive, Macro average rows
        for j in range(len(report_data[0])):
            if i <= 2:  # Class rows
                table[(i, j)].set_facecolor('#E8F5E8')
            elif i == 3:  # Macro average row
                table[(i, j)].set_facecolor('#FFF3CD')

    ax.set_title('Real-Data Detailed Performance Report', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/real_data_performance_report.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: real_data_performance_report.svg")

    # 9. Prediction error analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    error = real_pred != y_real
    correct = real_pred == y_real

    # Plot correct and incorrect predictions
    ax.scatter(np.where(correct)[0], real_prob[correct],
               alpha=0.5, label='Correct Prediction', color='green', s=100)
    ax.scatter(np.where(error)[0], real_prob[error],
               alpha=0.8, label='Incorrect Prediction', color='red', marker='x')

    # Add decision boundary
    ax.axhline(y=0.5, color='blue', linestyle='--', alpha=0.7, label='Decision Boundary (0.5)', linewidth=4)

    ax.set_xlabel('Sample Index', fontsize=28)
    ax.set_ylabel('Prediction Probability', fontsize=28)
    ax.set_title('Real-Data Prediction Analysis', fontsize=28, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(framealpha=True, fontsize=24)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/real_data_prediction_error.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: real_data_prediction_error.svg")

    # 10. Feature correlation heatmap for real data
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = feature_data.corr()
    sns.heatmap(corr_matrix, annot=True,
                cmap=sns.diverging_palette(220, 20, as_cmap=True),
                center=0, ax=ax,
                square=True, fmt='.2f',
                linewidths=0.6,
                linecolor='white',
                annot_kws={'fontsize': 10, 'fontweight': 'bold'},
                cbar_kws={'shrink': 0.8})
    ax.set_title('Real-Data Feature Correlation Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/real_data_feature_correlation.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: real_data_feature_correlation.svg")


# [Previous functions remain the same: generate_sample_data, load_excel_data, prepare_data, XGBoostDetailedModel class]
# ... (keeping all existing functions)

def plot_actual_vs_predicted(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Plot actual values versus predicted values for training, validation, and test sets
    This validates the accuracy of machine learning predictions
    """
    output_dir = "xgboost_plots"

    # Get predictions and probabilities
    train_pred, train_prob = model.predict(X_train)
    val_pred, val_prob = model.predict(X_val)
    test_pred, test_prob = model.predict(X_test)

    # Create comprehensive figure with subplots
    fig = plt.figure(figsize=(18, 12))

    # 1. Training Set: Actual vs Predicted (Scatter)
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_train, train_prob, alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=0.5)
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Decision Boundary')
    ax1.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Probabilities', fontsize=12, fontweight='bold')
    ax1.set_title(f'Training Set (n={len(y_train)})\nAUC={roc_auc_score(y_train, train_prob):.3f}',
                  fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)

    # 2. Validation Set: Actual vs Predicted (Scatter)
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(y_val, val_prob, alpha=0.6, s=50, c='green', edgecolors='black', linewidth=0.5)
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Decision Boundary')
    ax2.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted Probabilities', fontsize=12, fontweight='bold')
    ax2.set_title(f'Validation Set (n={len(y_val)})\nAUC={roc_auc_score(y_val, val_prob):.3f}',
                  fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)

    # 3. Test Set: Actual vs Predicted (Scatter)
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(y_test, test_prob, alpha=0.6, s=50, c='red', edgecolors='black', linewidth=0.5)
    ax3.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
    ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Decision Boundary')
    ax3.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Predicted Probabilities', fontsize=12, fontweight='bold')
    ax3.set_title(f'Test Set (n={len(y_test)})\nAUC={roc_auc_score(y_test, test_prob):.3f}',
                  fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(-0.1, 1.1)

    # 4. Prediction Error Distribution - Training
    ax4 = plt.subplot(2, 3, 4)
    train_errors = train_prob - y_train
    ax4.hist(train_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax4.axvline(x=np.mean(train_errors), color='green', linestyle='--', linewidth=2,
                label=f'Mean Error: {np.mean(train_errors):.3f}')
    ax4.set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title(f'Training Error Distribution\nMAE={np.mean(np.abs(train_errors)):.3f}',
                  fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Prediction Error Distribution - Validation
    ax5 = plt.subplot(2, 3, 5)
    val_errors = val_prob - y_val
    ax5.hist(val_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax5.axvline(x=np.mean(val_errors), color='green', linestyle='--', linewidth=2,
                label=f'Mean Error: {np.mean(val_errors):.3f}')
    ax5.set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax5.set_title(f'Validation Error Distribution\nMAE={np.mean(np.abs(val_errors)):.3f}',
                  fontsize=13, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Prediction Error Distribution - Test
    ax6 = plt.subplot(2, 3, 6)
    test_errors = test_prob - y_test
    ax6.hist(test_errors, bins=30, alpha=0.7, color='red', edgecolor='black')
    ax6.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax6.axvline(x=np.mean(test_errors), color='green', linestyle='--', linewidth=2,
                label=f'Mean Error: {np.mean(test_errors):.3f}')
    ax6.set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax6.set_title(f'Test Error Distribution\nMAE={np.mean(np.abs(test_errors)):.3f}',
                  fontsize=13, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Actual vs Predicted Values Analysis: Validation of ML Predictions',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/20_actual_vs_predicted_comprehensive.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 20_actual_vs_predicted_comprehensive.svg")

    # Additional plot: Combined scatter plot with all datasets
    fig, ax = plt.subplots(figsize=(12, 10))

    # Add jitter to separate overlapping points
    jitter_train = np.random.normal(0, 0.02, len(y_train))
    jitter_val = np.random.normal(0, 0.02, len(y_val))
    jitter_test = np.random.normal(0, 0.02, len(y_test))

    ax.scatter(y_train + jitter_train, train_prob, alpha=0.5, s=60, c='blue',
               label=f'Training (n={len(y_train)}, AUC={roc_auc_score(y_train, train_prob):.3f})',
               edgecolors='black', linewidth=0.5)
    ax.scatter(y_val + jitter_val, val_prob, alpha=0.5, s=60, c='green',
               label=f'Validation (n={len(y_val)}, AUC={roc_auc_score(y_val, val_prob):.3f})',
               edgecolors='black', linewidth=0.5)
    ax.scatter(y_test + jitter_test, test_prob, alpha=0.5, s=60, c='red',
               label=f'Test (n={len(y_test)}, AUC={roc_auc_score(y_test, test_prob):.3f})',
               edgecolors='black', linewidth=0.5)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=3, label='Perfect Prediction', alpha=0.7)
    ax.axhline(y=0.5, color='purple', linestyle='--', linewidth=2, alpha=0.5, label='Decision Boundary')

    ax.set_xlabel('Actual Values', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted Probabilities', fontsize=14, fontweight='bold')
    ax.set_title('Combined Actual vs Predicted: All Datasets', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/21_actual_vs_predicted_combined.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 21_actual_vs_predicted_combined.svg")


def plot_prediction_accuracy_metrics(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Create detailed accuracy metrics comparison table and visualization
    """
    output_dir = "xgboost_plots"

    # Get predictions
    train_pred, train_prob = model.predict(X_train)
    val_pred, val_prob = model.predict(X_val)
    test_pred, test_prob = model.predict(X_test)

    # Calculate metrics for each dataset
    datasets = ['Training', 'Validation', 'Test']
    y_true_list = [y_train, y_val, y_test]
    y_pred_list = [train_pred, val_pred, test_pred]
    y_prob_list = [train_prob, val_prob, test_prob]

    metrics_data = []
    for dataset, y_true, y_pred, y_prob in zip(datasets, y_true_list, y_pred_list, y_prob_list):
        metrics_data.append({
            'Dataset': dataset,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred),
            'AUC': roc_auc_score(y_true, y_prob),
            'AP': average_precision_score(y_true, y_prob),
            'MAE': np.mean(np.abs(y_prob - y_true))
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # 1. Bar chart comparison
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'AP']
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    for idx, dataset in enumerate(datasets):
        values = [metrics_df[metrics_df['Dataset'] == dataset][m].values[0] for m in metrics_to_plot]
        offset = (idx - 1) * width
        bars = ax1.bar(x + offset, values, width, label=dataset, alpha=0.8)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xlabel('Metrics', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax1.set_title('Performance Metrics Comparison Across Datasets', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_to_plot, rotation=0)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.1])

    # 2. Table with detailed metrics
    ax2.axis('tight')
    ax2.axis('off')

    table_data = []
    table_data.append(['Dataset', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'AP', 'MAE'])

    for _, row in metrics_df.iterrows():
        table_data.append([
            row['Dataset'],
            f"{row['Accuracy']:.4f}",
            f"{row['Precision']:.4f}",
            f"{row['Recall']:.4f}",
            f"{row['F1-Score']:.4f}",
            f"{row['AUC']:.4f}",
            f"{row['AP']:.4f}",
            f"{row['MAE']:.4f}"
        ])

    table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                      cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style the table
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i == 1:  # Training row
                table[(i, j)].set_facecolor('#E3F2FD')
            elif i == 2:  # Validation row
                table[(i, j)].set_facecolor('#E8F5E9')
            else:  # Test row
                table[(i, j)].set_facecolor('#FFEBEE')

    ax2.set_title('Detailed Metrics Table', fontsize=14, fontweight='bold', pad=20)

    plt.suptitle('Comprehensive Accuracy Validation Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/22_accuracy_metrics_comparison.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 22_accuracy_metrics_comparison.svg")

    return metrics_df


def plot_residual_analysis(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Plot residual analysis to validate prediction quality
    """
    output_dir = "xgboost_plots"

    # Get predictions
    train_pred, train_prob = model.predict(X_train)
    val_pred, val_prob = model.predict(X_val)
    test_pred, test_prob = model.predict(X_test)

    # Calculate residuals
    train_residuals = y_train - train_prob
    val_residuals = y_val - val_prob
    test_residuals = y_test - test_prob

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Residual plots
    # Training residuals
    axes[0, 0].scatter(train_prob, train_residuals, alpha=0.5, c='blue', s=30)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Probability', fontsize=18, fontweight='bold')
    axes[0, 0].set_ylabel('Residual (Actual - Predicted)', fontsize=18, fontweight='bold')
    axes[0, 0].set_title(
        f'Training Set Residuals\nMean={np.mean(train_residuals):.4f}, Std={np.std(train_residuals):.4f}',
        fontsize=18, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Validation residuals
    axes[0, 1].scatter(val_prob, val_residuals, alpha=0.5, c='green', s=30)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted Probability', fontsize=18, fontweight='bold')
    axes[0, 1].set_ylabel('Residual (Actual - Predicted)', fontsize=18, fontweight='bold')
    axes[0, 1].set_title(
        f'Validation Set Residuals\nMean={np.mean(val_residuals):.4f}, Std={np.std(val_residuals):.4f}',
        fontsize=18, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Test residuals
    axes[0, 2].scatter(test_prob, test_residuals, alpha=0.5, c='red', s=30)
    axes[0, 2].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 2].set_xlabel('Predicted Probability', fontsize=18, fontweight='bold')
    axes[0, 2].set_ylabel('Residual (Actual - Predicted)', fontsize=18, fontweight='bold')
    axes[0, 2].set_title(f'Test Set Residuals\nMean={np.mean(test_residuals):.4f}, Std={np.std(test_residuals):.4f}',
                         fontsize=18, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Q-Q plots (to check if residuals are normally distributed)
    from scipy import stats

    for idx, (residuals, title, color) in enumerate([
        (train_residuals, 'Training', 'blue'),
        (val_residuals, 'Validation', 'green'),
        (test_residuals, 'Test', 'red')
    ]):
        stats.probplot(residuals, dist="norm", plot=axes[1, idx])
        axes[1, idx].get_lines()[0].set_color(color)
        axes[1, idx].get_lines()[0].set_markersize(5)
        axes[1, idx].get_lines()[1].set_color('red')
        axes[1, idx].get_lines()[1].set_linewidth(2)
        axes[1, idx].set_title(f'{title} Set Q-Q Plot', fontsize=18, fontweight='bold')
        axes[1, idx].grid(True, alpha=0.3)

    plt.suptitle('Residual Analysis: Validation of Prediction Quality', fontsize=28, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/23_residual_analysis.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 23_residual_analysis.svg")


def plot_mae_learning_curve(X_train, y_train, X_val, y_val):
    """
    Plot MAE (Mean Absolute Error) Learning Curve
    Shows how MAE changes as training sample size increases
    """
    output_dir = "xgboost_plots"

    print("\nGenerating MAE Learning Curve...")

    # Define training sizes (from 10% to 100% of training data)
    train_sizes = np.linspace(0.1, 1.0, 15)

    train_mae_scores = []
    val_mae_scores = []
    train_mae_std = []
    val_mae_std = []
    actual_train_sizes = []

    # For each training size, train multiple models and calculate MAE
    for size in train_sizes:
        n_samples = int(size * len(X_train))
        if n_samples < 10:  # Skip if too few samples
            continue

        actual_train_sizes.append(n_samples)

        # Multiple runs for stability (5 runs with different random samples)
        temp_train_maes = []
        temp_val_maes = []

        for run in range(5):
            # Random subsample from training data
            indices = np.random.choice(len(X_train), n_samples, replace=False)
            X_sub = X_train[indices]
            y_sub = y_train[indices]

            # Train a temporary model
            temp_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42 + run,
                eval_metric='logloss',
                use_label_encoder=False
            )
            temp_model.fit(X_sub, y_sub, verbose=False)

            # Calculate MAE for training and validation
            train_pred_prob = temp_model.predict_proba(X_sub)[:, 1]
            val_pred_prob = temp_model.predict_proba(X_val)[:, 1]

            train_mae = np.mean(np.abs(y_sub - train_pred_prob))
            val_mae = np.mean(np.abs(y_val - val_pred_prob))

            temp_train_maes.append(train_mae)
            temp_val_maes.append(val_mae)

        # Calculate mean and std across runs
        train_mae_scores.append(np.mean(temp_train_maes))
        val_mae_scores.append(np.mean(temp_val_maes))
        train_mae_std.append(np.std(temp_train_maes))
        val_mae_std.append(np.std(temp_val_maes))

    # Convert to numpy arrays
    train_mae_scores = np.array(train_mae_scores)
    val_mae_scores = np.array(val_mae_scores)
    train_mae_std = np.array(train_mae_std)
    val_mae_std = np.array(val_mae_std)
    actual_train_sizes = np.array(actual_train_sizes)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot training MAE with confidence interval
    ax.plot(actual_train_sizes, train_mae_scores, 'o-', linewidth=3,
            markersize=8, color='blue', label='Training MAE')
    ax.fill_between(actual_train_sizes,
                    train_mae_scores - train_mae_std,
                    train_mae_scores + train_mae_std,
                    alpha=0.2, color='blue')

    # Plot validation MAE with confidence interval
    ax.plot(actual_train_sizes, val_mae_scores, 's-', linewidth=3,
            markersize=8, color='red', label='Validation MAE')
    ax.fill_between(actual_train_sizes,
                    val_mae_scores - val_mae_std,
                    val_mae_scores + val_mae_std,
                    alpha=0.2, color='red')

    # Add horizontal line for final validation MAE
    final_val_mae = val_mae_scores[-1]
    ax.axhline(y=final_val_mae, color='green', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Final Validation MAE: {final_val_mae:.4f}')

    # Styling
    ax.set_xlabel('Training Sample Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
    ax.set_title('MAE Learning Curve: Model Performance vs Training Size',
                 fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Add text annotation for interpretation
    gap = val_mae_scores[-1] - train_mae_scores[-1]
    if gap > 0.05:
        interpretation = "High variance (overfitting)"
        color = 'red'
    elif gap < 0.02:
        interpretation = "Good generalization"
        color = 'green'
    else:
        interpretation = "Moderate generalization"
        color = 'orange'

    ax.text(0.02, 0.98, f'Final Training MAE: {train_mae_scores[-1]:.4f}\n'
                        f'Final Validation MAE: {val_mae_scores[-1]:.4f}\n'
                        f'Gap: {gap:.4f}\n'
                        f'Assessment: {interpretation}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/24_mae_learning_curve.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 24_mae_learning_curve.svg")

    # Additional detailed plot with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: MAE Learning Curve (same as above)
    axes[0].plot(actual_train_sizes, train_mae_scores, 'o-', linewidth=3,
                 markersize=8, color='blue', label='Training MAE')
    axes[0].fill_between(actual_train_sizes,
                         train_mae_scores - train_mae_std,
                         train_mae_scores + train_mae_std,
                         alpha=0.2, color='blue')
    axes[0].plot(actual_train_sizes, val_mae_scores, 's-', linewidth=3,
                 markersize=8, color='red', label='Validation MAE')
    axes[0].fill_between(actual_train_sizes,
                         val_mae_scores - val_mae_std,
                         val_mae_scores + val_mae_std,
                         alpha=0.2, color='red')
    axes[0].set_xlabel('Training Sample Size', fontsize=18, fontweight='bold')
    axes[0].set_ylabel('Mean Absolute Error (MAE)', fontsize=18, fontweight='bold')
    axes[0].set_title('MAE Learning Curve', fontsize=18, fontweight='bold')
    # 左图图例：添加边框和不透明白色背景
    legend0 = axes[0].legend(fontsize=18, frameon=True, fancybox=True, shadow=True,
                             framealpha=1.0, facecolor='white', edgecolor='black',
                             loc='best')
    legend0.get_frame().set_linewidth(1.5)
    axes[0].grid(True, alpha=0.3)

    # Right: Gap between Training and Validation MAE
    mae_gap = val_mae_scores - train_mae_scores
    axes[1].plot(actual_train_sizes, mae_gap, 'o-', linewidth=3,
                 markersize=8, color='purple', label='Validation - Training MAE')
    axes[1].fill_between(actual_train_sizes, mae_gap, 0, alpha=0.3, color='purple')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes[1].axhline(y=0.05, color='red', linestyle='--', linewidth=2, alpha=0.5,
                    label='High Variance Threshold (0.05)')
    axes[1].set_xlabel('Training Sample Size', fontsize=18, fontweight='bold')
    axes[1].set_ylabel('MAE Gap (Validation - Training)', fontsize=18, fontweight='bold')
    axes[1].set_title('Overfitting Detection: MAE Gap Analysis', fontsize=18, fontweight='bold')
    # 右图图例：添加边框和不透明白色背景
    legend1 = axes[1].legend(fontsize=18, frameon=True, fancybox=True, shadow=True,
                             framealpha=1.0, facecolor='white', edgecolor='black',
                             loc='best')
    legend1.get_frame().set_linewidth(1.5)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Comprehensive MAE Learning Curve Analysis', fontsize=28, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/25_mae_learning_curve_detailed.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 25_mae_learning_curve_detailed.svg")

    # Print summary statistics
    print("\n=== MAE Learning Curve Summary ===")
    print(f"Initial Training MAE (10% data): {train_mae_scores[0]:.4f}")
    print(f"Final Training MAE (100% data): {train_mae_scores[-1]:.4f}")
    print(f"Initial Validation MAE: {val_mae_scores[0]:.4f}")
    print(f"Final Validation MAE: {val_mae_scores[-1]:.4f}")
    print(f"MAE Improvement (Validation): {val_mae_scores[0] - val_mae_scores[-1]:.4f}")
    print(f"Final MAE Gap: {gap:.4f}")
    print(f"Assessment: {interpretation}")

    return {
        'train_sizes': actual_train_sizes,
        'train_mae': train_mae_scores,
        'val_mae': val_mae_scores,
        'train_std': train_mae_std,
        'val_std': val_mae_std,
        'final_gap': gap
    }


# [Keep all other existing functions unchanged]

def main():
    """Main function with enhanced validation plots"""
    print("=== XGBoost Time Series Binary Classification Detailed Analysis ===\n")

    # [All existing main() code remains the same until visualization section]
    # ... (keep existing code)
    # 1. Load data
    print("1. Loading data...")
    X, y = load_excel_data('DATA-5.xlsx')

    if X is None or y is None:
        print("Excel data loading failed, using simulated data...")
        X, y = generate_sample_data()

    print(f"Data dimensions: X={X.shape}, y={y.shape}")
    print(
        f"Label distribution: Positive={np.sum(y)} ({100 * np.mean(y):.1f}%), Negative={len(y) - np.sum(y)} ({100 * (1 - np.mean(y)):.1f}%)")

    # 2. Data preprocessing
    print("\n2. Data preprocessing and splitting...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(X, y)

    # 3. Create and train XGBoost model
    print("\n3. Creating and training XGBoost model...")
    model = XGBoostDetailedModel()

    # Train model
    val_acc, val_auc, val_f1 = model.train(X_train, y_train, X_val, y_val)

    # 4. Cross validation
    print("\n4. Performing cross validation...")
    cv_scores = model.cross_validate(X_train, y_train, cv=5)

    # 5. Final test set evaluation
    print("\n5. Final test set evaluation...")
    test_pred, test_prob = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_prob)
    test_f1 = f1_score(y_test, test_pred)

    print(f"\n=== Final Test Results ===")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Cross-validation AUC Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, test_pred,
                                target_names=['Negative', 'Positive']))

    # 6. Comprehensive visualization analysis
    print("\n6. Generating detailed visualization analysis...")

    # Main analysis charts
    print("Plotting comprehensive analysis charts...")
    results = save_individual_plots(model, X_train, y_train, X_val, y_val, X_test, y_test)

    # Training process analysis
    print("Plotting training process analysis...")
    plot_training_progress(model)

    # Feature analysis
    print("Plotting feature interaction analysis...")
    if model.feature_importance is not None:
        analyze_feature_interactions(model, X_train, model.feature_names)
    else:
        print("Skipping feature analysis (feature importance data unavailable)")

    # Performance report table
    print("Creating performance report table...")
    create_performance_report_table(y_test, test_pred, test_prob)
    # After section 6, add new validation plots:
    print("\n7. Generating additional validation plots...")

    print("Plotting actual vs predicted values...")
    plot_actual_vs_predicted(model, X_train, y_train, X_val, y_val, X_test, y_test)

    print("Creating accuracy metrics comparison...")
    metrics_comparison = plot_prediction_accuracy_metrics(model, X_train, y_train, X_val, y_val, X_test, y_test)
    print("\nMetrics Comparison Summary:")
    print(metrics_comparison)

    print("Performing residual analysis...")
    plot_residual_analysis(model, X_train, y_train, X_val, y_val, X_test, y_test)
    print("Performing mae_learning_curve...")
    plot_mae_learning_curve(X_train, y_train, X_val, y_val)

    # [Rest of existing code continues...]
    # 7. Model interpretation and recommendations
    print("\n=== XGBoost Model Analysis Summary ===")
    print(
        f"✓ Model achieved AUC of {test_auc:.3f} on test set, performance is {'Excellent' if test_auc > 0.8 else 'Good' if test_auc > 0.7 else 'Fair'}")
    print(f"✓ Cross-validation results are stable, AUC standard deviation is {cv_scores.std():.3f}")

    if model.feature_importance is not None:
        most_important = model.feature_names[np.argmax(model.feature_importance)]
        print(f"✓ Most important feature is {most_important}, importance: {np.max(model.feature_importance):.3f}")

        # Feature importance explanation
        importance_ranking = sorted(zip(model.feature_names, model.feature_importance),
                                    key=lambda x: x[1], reverse=True)
        print("\nFeature Importance Ranking:")
        for i, (feature, importance) in enumerate(importance_ranking, 1):
            print(f"{i}. {feature}: {importance:.3f}")
    else:
        print("✗ Unable to obtain feature importance information")

    print(f"\nRecommend using this XGBoost model for time series binary classification prediction")

    # 8. Validate Real-Data.xlsx
    print("\n8. Validating Real-Data.xlsx...")
    real_data_results = validate_real_data(model, scaler)

    print("\n=== Real-Data Validation Complete ===")
    print(f"All plots have been saved in 'real_data_plots' directory")
    return model, results, real_data_results


if __name__ == "__main__":
    model, results, real_data_results = main()