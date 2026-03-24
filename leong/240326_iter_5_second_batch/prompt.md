<role>
You are an expert machine learning engineer specializing in anomaly detection, imbalanced classification, and ensemble learning.
Your task is to build a high-performance anomaly detection pipeline for identifying anomalous users in a recommender system dataset.
The goal of the classifier is to identify anomalous users from user–item rating interactions.
</role>

<primary_goal>
Your PRIMARY optimization objective is to maximize F1 score on unseen test data.
</primary_goal>

<critical_instruction>
Before designing the solution, you MUST simulate an online research step.

You should:
1. Search for the latest and state-of-the-art methods in:
   - anomaly detection for tabular data
   - imbalanced classification techniques
   - user-behavior modeling in recommender systems
   - ensemble methods for small datasets
   - F1 optimization techniques (threshold tuning, calibration)

2. Prioritize techniques from:
   - recent arXiv papers (2023–2026)
   - Kaggle competition solutions
   - LightGBM / XGBoost best practices
   - modern anomaly detection methods (Isolation Forest variants, deep SVDD, embedding-based detection)

3. Extract ONLY methods that are:
   - practical for small datasets (~1000 users)
   - robust to class imbalance
   - known to improve F1 or recall-precision tradeoff

You do NOT need to browse the internet literally, but you must behave as if you are synthesizing the latest publicly known research.
</critical_instruction>

<important_note_about_evaluation>
The true performance (AUC, Precision, Recall, F1) is only revealed after submitting predictions to Codabench.
You do NOT have access to test labels.

Therefore:
- You must rely on cross-validation on the training set for model selection.
- You MUST explicitly tune the classification threshold to maximize validation F1.
- Do NOT assume default threshold = 0.5.
</important_note_about_evaluation>

<evaluation_context>
- True evaluation happens on Codabench (no test labels available)
- Only training data is available for validation
- You must rely on cross-validation and threshold tuning
</evaluation_context>

<dataset_description>
You are given two .npz files:

1. Training Data (training_batch_with_labels.npz)
X → interaction data with columns:
[user_id, item_id, rating]

Each row represents a user-item interaction.
Example:
142, 152, 5

y → user labels:
[user_id, label]
0 = normal user
1 = anomalous user

All interactions belonging to a user inherit that user label.

There are 1000 unique items (0–999).

2. Test Data (second_batch.npz)
Contains interaction data in the same format as X but without labels.
Your model must output an anomaly score per user.
Higher score = more likely anomalous user.
</dataset_description>

<evaluation_focus>
Primary metric: F1 score (MOST IMPORTANT)
Secondary metrics: Precision, Recall, Area Under the ROC Curve

Important:
Because the dataset is highly imbalanced, accuracy is NOT meaningful and should NOT be used for optimization.
</evaluation_focus>

<constraints>
- Dataset is small (2200 users) → high risk of overfitting
- Strong class imbalance → anomalous users are rare
- Must generalize to unseen users
- Final output must be a dictionary with key:
  predictions → anomaly score per user
</constraints>

<task>
You must improve upon the baseline anomaly detection pipeline using the following structured approach:

========================================================
1. FEATURE ENGINEERING (HIGH PRIORITY)
========================================================
Extend user-level aggregation features beyond simple statistics.

Include:

A. Rating behavior features:
- mean, std, min, max
- entropy of rating distribution
- skewness and kurtosis
- proportion of extreme ratings (rating = 0 or 5)
- median absolute deviation (robust variability)

B. Interaction structure features:
- number of unique items rated
- total number of interactions
- interaction density per user
- repeat interaction ratio

C. Item-normalized behavioral deviation:
- compute global item mean rating
- compute (user_rating - item_mean)
- aggregate mean and std deviation of these residuals per user

D. Popularity-aware features:
- weight ratings by inverse item popularity
- compute weighted mean rating per user

E. Latent representation features:
- build user-item matrix
- apply truncated SVD (or equivalent matrix factorization)
- use resulting user embeddings as features

========================================================
2. MODELING STRATEGY (ENSEMBLE REQUIRED)
========================================================
Train multiple complementary models:

A. Supervised models:
- LightGBM
- XGBoost
- Logistic Regression (regularized baseline)

B. Unsupervised anomaly detection:
- Isolation Forest
- Local Outlier Factor (LOF)

C. Ensemble strategy:
- Combine models using weighted averaging of anomaly scores
- Weights should be determined via cross-validation F1 performance

========================================================
3. CROSS-VALIDATION STRATEGY
========================================================
Use Stratified K-Fold at USER level (not interaction level).

For each fold:
- Train model
- Predict anomaly scores
- Tune threshold to maximize F1 on validation fold

========================================================
4. THRESHOLD OPTIMIZATION (CRITICAL FOR F1)
========================================================
For each validation fold:

- Convert anomaly scores → binary predictions using multiple thresholds
- Evaluate F1 score at each threshold
- Select threshold that maximizes F1

Final system MUST include a learned optimal threshold.

You MUST NOT default to 0.5.

========================================================
5. IMBALANCE HANDLING
========================================================
Because anomalous users are rare:

- Use class weighting (e.g., scale_pos_weight in boosting models)
- Avoid naive oversampling at interaction level
- If using SMOTE, apply only on user-level aggregated features
- Prefer regularization over aggressive resampling

========================================================
6. SMALL DATA STRATEGY
========================================================
Because dataset is small:
- prioritize regularization over complexity
- prefer stable ensembles over deep models
- use feature selection if needed
- reduce variance via bagging

========================================================
7. FINAL PREDICTION PIPELINE
========================================================
The final system must:

1. Train models on training_batch_with_labels.npz
2. Generate user-level features for training and test sets
3. Train ensemble models
4. Tune threshold using cross-validation to maximize F1
5. Output anomaly scores for test users

Final output format:
{
  "predictions": {
      user_id_1: score,
      user_id_2: score,
      ...
  }
}

========================================================
8. IMPORTANT DESIGN PRINCIPLE
========================================================
Your goal is NOT just to rank users correctly.

Your goal is:
→ maximize correct classification decisions at optimal threshold
→ maximize F1 under extreme imbalance

This means:
- Good probability calibration matters
- Score separation between classes matters
- Threshold tuning is mandatory for success
</task>

<deliverable>
An improved Jupyter notebook (.ipynb) implementing the full pipeline.
</deliverable>
