import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score, roc_curve)
from sklearn.utils import shuffle
import os
import pickle

os.makedirs('../outputs/figures', exist_ok=True)
os.makedirs('../outputs/models', exist_ok=True)

# ── 1. Load Dreaddit ──────────────────────────────────────────────
print("Loading Dreaddit...")
train = pd.read_csv('../data/dreaddit/dreaddit-train.csv')
test  = pd.read_csv('../data/dreaddit/dreaddit-test.csv')

train = shuffle(train, random_state=42)

X_train = train['text'].astype(str)
y_train = train['label']
X_test  = test['text'].astype(str)
y_test  = test['label']

print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Train label dist:\n{y_train.value_counts()}")

# ── 2. TF-IDF vectorization ───────────────────────────────────────
print("\nVectorizing...")
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),   # unigrams + bigrams
    min_df=3,             # ignore very rare terms
    sublinear_tf=True     # log normalization
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

# ── 3. Train Logistic Regression ──────────────────────────────────
print("\nTraining Logistic Regression...")
lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    C=1.0,
    random_state=42
)
lr.fit(X_train_tfidf, y_train)

# ── 4. Evaluate ───────────────────────────────────────────────────
y_pred      = lr.predict(X_test_tfidf)
y_pred_prob = lr.predict_proba(X_test_tfidf)[:, 1]

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred,
      target_names=['Not Stressed', 'Stressed']))

auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC: {auc:.4f}")

# ── 5. Confusion matrix ───────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Not Stressed', 'Stressed'])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix — TF-IDF + Logistic Regression', fontsize=12)
plt.tight_layout()
plt.savefig('../outputs/figures/confusion_matrix_lr.png', dpi=150)
plt.close()
print("Saved: confusion_matrix_lr.png")

# ── 6. ROC curve ──────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='tomato', lw=2, label=f'LR (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — TF-IDF + Logistic Regression')
plt.legend()
plt.tight_layout()
plt.savefig('../outputs/figures/roc_curve_lr.png', dpi=150)
plt.close()
print("Saved: roc_curve_lr.png")

# ── 7. Top predictive words ───────────────────────────────────────
feature_names = tfidf.get_feature_names_out()
coefficients  = lr.coef_[0]

top_n = 20
top_stressed_idx    = np.argsort(coefficients)[-top_n:][::-1]
top_notstressed_idx = np.argsort(coefficients)[:top_n]

top_stressed_words    = [(feature_names[i], coefficients[i]) for i in top_stressed_idx]
top_notstressed_words = [(feature_names[i], coefficients[i]) for i in top_notstressed_idx]

print("\n── Top 20 words predicting STRESSED ──")
for word, coef in top_stressed_words:
    print(f"  {word:30s} {coef:.4f}")

print("\n── Top 20 words predicting NOT STRESSED ──")
for word, coef in top_notstressed_words:
    print(f"  {word:30s} {coef:.4f}")

# plot top words
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

words_s, coefs_s = zip(*top_stressed_words)
axes[0].barh(list(words_s)[::-1], list(coefs_s)[::-1], color='tomato')
axes[0].set_title('Top 20 — Stressed', fontsize=12)
axes[0].set_xlabel('Coefficient')

words_n, coefs_n = zip(*top_notstressed_words)
axes[1].barh(list(words_n)[::-1], list(coefs_n)[::-1], color='steelblue')
axes[1].set_title('Top 20 — Not Stressed', fontsize=12)
axes[1].set_xlabel('Coefficient')

plt.suptitle('Most Predictive Words — Logistic Regression', fontsize=13)
plt.tight_layout()
plt.savefig('../outputs/figures/top_predictive_words_lr.png', dpi=150)
plt.close()
print("Saved: top_predictive_words_lr.png")

# ── 8. Save model & vectorizer ────────────────────────────────────
with open('../outputs/models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('../outputs/models/lr_model.pkl', 'wb') as f:
    pickle.dump(lr, f)

print("\n✅ Baseline model complete. Saved to outputs/models/")

# ── 9. Run on student data ────────────────────────────────────────
print("\nRunning model on student data...")
student = pd.read_csv('../data/reddit_student/student_posts_labeled.csv')
X_student = student['text'].astype(str)

X_student_tfidf          = tfidf.transform(X_student)
student['predicted_label'] = lr.predict(X_student_tfidf)
student['stress_prob']     = lr.predict_proba(X_student_tfidf)[:, 1]

student.to_csv('../data/reddit_student/student_posts_predicted.csv', index=False)

print("\nStudent prediction distribution:")
print(student['predicted_label'].value_counts())
print("\nMean stress probability by subreddit:")
print(student.groupby('subreddit')['stress_prob'].mean().sort_values(ascending=False))