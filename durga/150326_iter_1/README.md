# CS421 Anomaly Detection — Semi-Supervised Approach
---

## Overview

This project identifies anomalous users in a recommender system dataset where users rate items on a 0–5 star scale. Approximately 9% of users are anomalous. The pipeline uses a semi-supervised approach. Models are trained on normal-only users and evaluated on all 1100 users and are combined with supervised models trained via 5-fold stratified cross-validation.

---

## Pipeline Summary

### 1. Feature Engineering (38 features per user)

**Rating statistics**: mean, std, variance, min/max, star fractions, entropy, skewness, bimodality, cosine deviation from normal-user centroid

**Item behaviour**: number of interactions, unique items, item ID gaps, sequential run fraction, item-rating autocorrelation, burst score, repeat item fraction

**Attack-profile features**: engineered to detect known recommender system attack patterns:
- `avg_attack_score`: detects average attackers who rate close to item means
- `bandwagon_score`: detects users whose ratings correlate with item popularity
- `random_attack_score`: detects random attackers via normalised rating entropy
- `segment_score`: detects concentration on popular items with extreme ratings
- `love_hate_score`: detects push/nuke attackers who give only 0s or 5s
- `rating_deviation_std`: measures consistency of deviation from item averages

Features with individual AUC < 0.55 are dropped. Remaining features are split into a rating view and an item view for model diversity. The scaler is fit on normal-only users to prevent leakage.

### 2. Models

**Semi-supervised (trained on normal users only):**
- GMM: full features, rating view, item view (3 separate models)
- Isolation Forest
- One-Class SVM
- LOF (Local Outlier Factor)
- HBOS (Histogram-Based Outlier Score)
- Denoising Autoencoder: adds Gaussian noise during training to learn a tighter normal manifold
- VAE: probabilistic reconstruction; anomaly score = reconstruction error + KL divergence

**Supervised (5-fold stratified CV):**
- LightGBM: regularised (num_leaves=15, L1/L2 penalty)
- XGBoost: regularised (max_depth=3, min_child_weight=5)
- Logistic Regression
- Stacking meta-learner: trained on out-of-fold predictions of all base models

### 3. Adaptive Ensemble

- Only methods with AUC ≥ 0.65 are included
- Weight per method = (AUC − 0.65) / sum of all excesses
- Supervised methods are penalised ×0.5 if their CV AUC < 0.80, preventing overfitted supervised models from dominating when anomaly patterns shift

### 4. Score Stretching

Raw ensemble scores are rank-normalised then raised to the power 0.4. This spreads scores to fully use the [0, 1] range, ensuring the most anomalous users score near 1.0.

### 5. Iterative Retraining

Each test batch is saved, and re-running the notebook automatically stacks it onto the training data, giving the supervised models direct signal about the current anomaly generation pattern.

---

## 3 Different Submission Strategy

| Submission | Contents | Best when |
|------------|----------|-----------|
| Sub 1 | Semi-supervised ensemble only | Anomaly pattern shifts week to week |
| Sub 2 | Full adaptive ensemble | Anomaly pattern is stable |
| Sub 3 | Best single model (LightGBM) | Supervised models are clearly strongest |

---

## Evaluation

The primary metric is **AUC (Area Under the ROC Curve)**. Precision and F1 scores reported by the scoring script use an arbitrary 0.5 threshold and are not meaningful for this task — the optimal threshold computed on training data yields F1 ≈ 0.65 for LightGBM.
