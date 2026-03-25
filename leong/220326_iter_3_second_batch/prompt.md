You are an expert ML engineer specializing in imbalanced anomaly detection for recommender systems.

You are given iter2 of an anomaly detection pipeline (described in full below) that achieves:
  OOF (training)  → AUC: 0.9448 | Precision: 0.875 | Recall: 0.56  | F1: 0.6829
  Codabench (test) → AUC: 0.6793 | Precision: 0.182 | Recall: 0.033 | F1: 0.0563

The enormous gap between OOF F1 (0.68) and Codabench F1 (0.056) is the central problem.
This is a catastrophic overfitting / distribution-shift failure — NOT a threshold issue.
Your sole objective is to produce a complete, runnable iter3 Jupyter notebook (.ipynb) that 
maximizes F1 on the unseen Codabench test set.

════════════════════════════════════════════════════════════════
ROOT CAUSE ANALYSIS — Address all of these:
════════════════════════════════════════════════════════════════

1. OVERFIT MODELS: RandomForest/ExtraTrees/GradientBoosting with 400 estimators on 
   ~1100 users memorize training patterns. The test set users are OOD (out-of-distribution).
   → Fix: Use LightGBM with aggressive regularization (min_child_samples≥20, reg_lambda≥5, 
     num_leaves≤15). Drop ExtraTrees entirely. Keep RF only as a diversity component.

2. NAIVE OVERSAMPLING LEAKS SIGNAL: Random duplication of anomalous users inside CV folds 
   inflates OOF scores without improving generalization.
   → Fix: Replace with SMOTE (from imblearn) or class_weight alone. 
     If SMOTE, apply only within the training fold after scaling.

3. ENSEMBLE WEIGHTS OPTIMIZED FOR AUC, NOT F1: The Nelder-Mead step minimizes -AUC.
   The stated goal is F1. This is a direct misalignment.
   → Fix: Change neg_auc_blend() to neg_f1_blend() — optimize for F1 at the threshold 
     jointly found during weight search.

4. SVD FEATURES OVERFIT ON SMALL DATA: 15 SVD components on ~1100 users with 10:1 
   imbalance is too expressive. These components capture training-specific noise.
   → Fix: Reduce to SVD_COMPONENTS = 5. Add SVD reconstruction error per user as an 
     unsupervised anomaly signal instead of raw components. Drop raw SVD components 
     from the final feature set (or reduce to top 3 by explained variance).

5. NO UNSUPERVISED ANOMALY SIGNALS: IsolationForest and LOF are imported but unused.
   These models generalize better to unseen users because they don't rely on labels.
   → Fix: Add IsolationForest score and LOF score as two additional features BEFORE 
     training the supervised classifiers. Fit both on the full training feature set.

6. THRESHOLD SELECTED ON FULL OOF, NOT CROSS-VALIDATED: The current threshold (0.553) 
   is biased because it's selected on the same OOF predictions used for blending weights.
   → Fix: Use nested CV or a held-out 20% stratified split solely for threshold tuning.
     Report threshold standard deviation across folds to assess stability.

7. MISSING FEATURE: No rating-count normalization relative to the user population.
   Anomalous users often have abnormally high or low interaction counts.
   → Fix: Add z-score of num_ratings (standardized across all users in training) as an 
     explicit feature. Also add percentile rank of num_ratings.

════════════════════════════════════════════════════════════════
DATASET FACTS (do not change these):
════════════════════════════════════════════════════════════════
- Training: 1100 users (1000 normal, 100 anomalous) → 10:1 imbalance
- Interactions: ~177k rows with [user_id, item_id, rating]  
- Rating range: 0–5, Items: 1000 unique
- Files: training_batch_with_labels.npz (keys: X, y), first_batch.npz (key: X)
- Submission: np.savez("submission_batch1.npz", predictions=scores_array)
  where scores_array is ordered by the user order in first_batch.npz

════════════════════════════════════════════════════════════════
REQUIRED CHANGES — implement ALL of the following:
════════════════════════════════════════════════════════════════

SECTION 1 — IMPORTS
- Add: lightgbm, imblearn.over_sampling.SMOTE
- Keep all existing imports

SECTION 2 — FEATURE ENGINEERING (build_features function)
Keep all existing feature groups PLUS add:
  a. svd_reconstruction_error: 
       fit TruncatedSVD(n_components=5), reconstruct matrix, compute per-user 
       mean squared reconstruction error → strong unsupervised anomaly signal
  b. rating_count_zscore: z-score of num_ratings across all users in the dataset
  c. rating_count_percentile: percentile rank of num_ratings (use scipy.stats.percentileofscore)
  d. DROP raw svd_0…svd_14 from the feature DataFrame (use reconstruction_error only)
  e. per_item_zscore_mean: for each user, compute mean of z-scored ratings 
       (z-score per item across all users), then take the user-level mean.
       → anomalous users systematically give ratings that are statistical outliers per item.

SECTION 3 — UNSUPERVISED ANOMALY SCORES AS META-FEATURES
After building features, BEFORE the CV loop, add:
  a. Fit IsolationForest(n_estimators=200, contamination=0.09, random_state=42) 
     on X_arr (StandardScaler-transformed). Add score_samples() output as feature 
     "iso_score" (note: more negative = more anomalous, so multiply by -1).
  b. Fit LocalOutlierFactor(n_neighbors=20, contamination=0.09, novelty=True) on X_arr.
     Add negative_outlier_factor_ as feature "lof_score".
  These must be fitted on training data only and applied to test data 
  using transform() / predict() — NO re-fitting on test.

SECTION 4 — MODEL CONFIGS
Replace the three models with:
  a. LightGBM (PRIMARY — weight 0.5):
       lgb.LGBMClassifier(
           n_estimators=500, num_leaves=15, max_depth=4,
           learning_rate=0.03, min_child_samples=20,
           reg_alpha=1.0, reg_lambda=5.0,
           scale_pos_weight=10,   # ratio of negatives to positives
           subsample=0.7, colsample_bytree=0.7,
           random_state=42, n_jobs=-1
       )
  b. RandomForest (SECONDARY — weight 0.3):
       RandomForestClassifier(
           n_estimators=300, max_depth=7, min_samples_leaf=5,
           class_weight="balanced", max_features="sqrt",
           random_state=42, n_jobs=-1
       )
  c. LogisticRegression (REGULARIZED BASELINE — weight 0.2):
       LogisticRegression(
           C=0.1, class_weight="balanced", 
           solver="lbfgs", max_iter=1000, random_state=42
       )

SECTION 5 — CLASS IMBALANCE INSIDE CV FOLDS
Replace random oversampling with SMOTE:
  from imblearn.over_sampling import SMOTE
  smote = SMOTE(k_neighbors=3, random_state=42 + fold_idx)  
  # k_neighbors=3 because minority class is small
  X_tr_aug, y_tr_aug = smote.fit_resample(X_tr_s, y_tr)
  # Apply SMOTE AFTER scaling, only on training portion of each fold

SECTION 6 — ENSEMBLE WEIGHT OPTIMIZATION (CRITICAL FIX)
Change the objective from AUC to F1:
  def neg_f1_blend(weights, oof_list, y):
      w = np.maximum(weights, 0)
      w = w / (w.sum() + 1e-9)
      blend = sum(wi * oi for wi, oi in zip(w, oof_list))
      # Search threshold jointly
      best_f1 = 0
      for t in np.linspace(0.05, 0.95, 200):
          f1 = f1_score(y, (blend >= t).astype(int), zero_division=0)
          if f1 > best_f1:
              best_f1 = f1
      return -best_f1

SECTION 7 — THRESHOLD TUNING (CROSS-VALIDATED)
Instead of selecting threshold on the full OOF predictions:
  a. Collect per-fold best thresholds during CV (threshold that maximizes F1 on 
     each fold's validation set).
  b. Final threshold = median of per-fold best thresholds.
  c. Print threshold mean ± std to show stability.
  d. Apply this threshold to final_scores for an optional binary output 
     (but still save continuous scores as predictions).

SECTION 8 — FINAL TEST PIPELINE
No changes needed except:
  - Apply the iso_score and lof_score transformations to test features 
    using the models fitted on training data (novelty=True for LOF).
  - Ensure feature column order matches exactly between train and test.

════════════════════════════════════════════════════════════════
EXPLICIT ANTI-PATTERNS — do NOT do these:
════════════════════════════════════════════════════════════════
- Do NOT fit SVD, IsolationForest, LOF, or StandardScaler on test data
- Do NOT select threshold on the same data used to optimize ensemble weights
- Do NOT use n_estimators > 500 or num_leaves > 31 (overfitting risk)
- Do NOT include ExtraTrees (too correlated with RF, adds no diversity)
- Do NOT use raw SVD components as features (use reconstruction error only)
- Do NOT apply SMOTE to validation folds, only training folds
- Do NOT optimize ensemble weights using AUC

════════════════════════════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════════════════════════════
Return a complete Jupyter notebook with these sections:
  1. Imports & Config
  2. Data Loading & EDA
  3. Feature Engineering (build_features function, fully updated)
  4. Unsupervised Meta-Features (IsoForest + LOF)
  5. CV Training Loop (LightGBM + RF + LogReg, SMOTE per fold)
  6. Ensemble Weight Optimization (F1-targeted)
  7. Threshold Cross-Validation
  8. Full Evaluation Report (OOF metrics + baseline comparison table)
  9. Test Prediction & Submission

Each section must have:
- A markdown cell explaining the design decision and WHY it improves generalization
- Inline comments on non-obvious lines
- Print statements showing key metrics at each step

The notebook must be self-contained and runnable top-to-bottom on Google Colab 
after mounting Google Drive and changing directory to where the .npz files are stored.