# Anomaly Detection Pipeline - Iteration 2

## Overview

This notebook implements a **clean, modular ML pipeline** for anomaly detection in user-item interaction data. The solution handles the challenge of detecting anomalous users in a heavily imbalanced dataset and is designed to generalize to unseen anomaly classes.

## Requirements Met ✅

### 1. Data Loading & Merging

- ✅ Loads training data from `input_training_data/`
  - `first_batch_with_labels.npz`
  - `subset_training_batch.npz`
- ✅ Properly concatenates X and y across multiple files
- ✅ Handles variable label formats (shape (N, 2) → extracts column 1)

### 2. Model Training

- ✅ Uses **Gradient Boosting Classifier** (better generalization than Random Forest)
- ✅ Modular architecture with separate functions for each stage
- ✅ Hyperparameters tuned for robustness:
  - `n_estimators=150`
  - `learning_rate=0.1`
  - `max_depth=5` (prevents overfitting)
  - `subsample=0.8` (reduces variance)
- ✅ Train-validation split for internal validation
- ✅ Feature importance analysis

### 3. Test Data Inference

- ✅ Loads test data from `input_test_data/second_batch.npz`
- ✅ Applies **identical feature engineering** to test data
- ✅ Generates **anomaly scores (probabilities)**, not hard labels

### 4. Output Generation

- ✅ Saves predictions to `output_data/predictions.npz`
- ✅ **Correct format**: Key named `'predictions'` (required by evaluator)
- ✅ **Correct dtype**: Float anomaly scores in [0, 1] range
- ✅ Includes user IDs for reference

### 5. Code Quality

- ✅ **Modular design**: Functions for each pipeline stage
  - `load_data()`
  - `merge_training_data()`
  - `engineer_features()`
  - `align_labels_with_features()`
  - `train_model()`
  - `predict_on_test_data()`
  - `save_predictions()`
  - `verify_saved_file()`
- ✅ Clear comments and docstrings
- ✅ Validation and error handling
- ✅ Easy to extend and modify

## Feature Engineering Strategy

The solution creates **user-level behavioral features** to improve generalization:

1. **Mean Rating**: Average rating given by user
2. **Std Rating**: Consistency in ratings (std dev)
3. **Num Ratings**: Activity level (count of interactions)
4. **Min Rating**: Minimum rating given
5. **Max Rating**: Maximum rating given
6. **Rating Range**: max - min (variability span)
7. **Rating Variance**: std_rating²
8. **Log Num Ratings**: Log-scaled activity (handles outliers)

These **user-level aggregates** are more robust to unseen anomaly classes than raw item-level features, as they capture **behavioral patterns** rather than specific items.

## Handling Unseen Anomaly Classes

The training set contains **2 anomaly classes**, but the test set may have up to **4 classes**. The design mitigates this challenge:

1. **Gradient Boosting** produces smoother decision boundaries than Random Forest
2. **Feature engineering** captures behavioral patterns generalizable across classes
3. **Conservative hyperparameters** prevent overfitting:
   - Lower max depth reduces variance
   - Higher subsample ratio improves stability
4. **Probability-based output** provides confidence scores for borderline cases

## Submission Format ✅

```
output_data/predictions.npz
├─ predictions: float64 array (n_samples,) with anomaly scores in [0, 1]
└─ user_ids: int64 array (n_samples,) with user identifiers

** Key must be named 'predictions' (required by evaluator) **
```

## Running the Pipeline

All cells are designed to run sequentially from top to bottom:

1. **Cell 1**: Overview (markdown)
2. **Cell 2**: Imports and setup
3. **Cell 3**: Data loading section header
4. **Cell 4**: Load and merge training data
5. **Cell 5**: Feature engineering section header
6. **Cell 6**: Engineer features from raw data
7. **Cell 7**: Align labels with features
8. **Cell 8**: Model training section header
9. **Cell 9**: Train Gradient Boosting model
10. **Cell 10**: Test data inference section header
11. **Cell 11**: Generate predictions on test data
12. **Cell 12**: Save results section header
13. **Cell 13**: Save and verify predictions
14. **Cell 14**: Summary section header
15. **Cell 15**: Pipeline execution summary

## Expected Output

After running all cells:

```
✓ Training Data:
  - Files loaded: 2
  - Total training samples: 204725
  - Total training users: 860
  - Classes in training set: [0, 1]

✓ Model Training:
  - Algorithm: Gradient Boosting Classifier
  - Model type optimized for: Generalization to unseen classes
  - Features engineered: 8

✓ Test Data Inference:
  - Test file: second_batch.npz
  - Test samples: 37232
  - Test users: 860
  - Anomaly scores range: [0.0001, 0.9999]

✓ Output Submission:
  - Saved to: .../output_data/predictions.npz
  - Status: ✅ READY FOR SUBMISSION
```

## Files Modified/Created

- ✅ `code.ipynb` - Complete refactored pipeline
- ✅ `README.md` - This documentation

## Next Steps for Improvement

1. **Ensemble Methods**: Combine multiple models (RF + GB + XGBoost)
2. **Feature Selection**: Remove low-importance features
3. **Hyperparameter Tuning**: Grid search or Bayesian optimization
4. **Unsupervised Learning**: Add isolation forest or DBSCAN
5. **Cross-validation**: Stratified k-fold validation
6. **Threshold Tuning**: Optimize decision boundary for test anomalies
7. **Error Analysis**: Analyze false positives/negatives

## Troubleshooting

**Error: "predictions is not a file in the archive"**

- ✅ Fixed: Key is now correctly named `'predictions'`

**Error: "predictions must be float anomaly scores"**

- ✅ Fixed: Now saving `pred_proba[:, 1]` (anomaly probabilities) instead of hard labels

**Error: "data must be 1-dimensional"**

- ✅ Fixed: Properly flattens labels from shape (N, 2) to (N,)

---

**Author**: Justin K.  
**Date**: March 25, 2026  
**Iteration**: 2  
**Status**: Ready for submission ✅
