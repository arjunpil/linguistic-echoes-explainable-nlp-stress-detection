import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import nltk
import os
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

os.makedirs('../../outputs/figures', exist_ok=True)
os.makedirs('../../outputs', exist_ok=True)

STOP_WORDS = set(stopwords.words('english'))

# ── 1. Load model & data ──────────────────────────────────────────
print("Loading model and data...")
with open('../../outputs/models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('../../outputs/models/lr_model.pkl', 'rb') as f:
    lr = pickle.load(f)

test    = pd.read_csv('../../data/dreaddit/dreaddit-test.csv')
student = pd.read_csv('../../data/reddit_student/student_posts_predicted.csv')

X_test = test['text'].astype(str)
y_test = test['label']

X_test_tfidf      = tfidf.transform(X_test)
test['predicted']  = lr.predict(X_test_tfidf)
test['stress_prob'] = lr.predict_proba(X_test_tfidf)[:, 1]

# ── 2. Error categorization ───────────────────────────────────────
test['error_type'] = 'correct'
test.loc[(test['label'] == 1) & (test['predicted'] == 0), 'error_type'] = 'false_negative'
test.loc[(test['label'] == 0) & (test['predicted'] == 1), 'error_type'] = 'false_positive'

error_counts = test['error_type'].value_counts()
print("\nError breakdown:")
print(error_counts)

plt.figure(figsize=(6, 4))
colors = {'correct': 'steelblue', 'false_negative': 'tomato', 'false_positive': 'orange'}
bars = plt.bar(error_counts.index, error_counts.values,
               color=[colors[k] for k in error_counts.index])
plt.title('Prediction Error Breakdown — LR Baseline', fontsize=12)
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('../../outputs/figures/error_breakdown.png', dpi=150)
plt.close()
print("Saved: error_breakdown.png")

# ── 3. Text length vs error type ─────────────────────────────────
test['word_count'] = test['text'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(7, 4))
sns.boxplot(x='error_type', y='word_count', data=test,
            order=['correct', 'false_positive', 'false_negative'],
            palette=['steelblue', 'orange', 'tomato'])
plt.title('Word Count by Prediction Outcome', fontsize=12)
plt.xlabel('Prediction Type')
plt.ylabel('Word Count')
plt.ylim(0, 400)
plt.tight_layout()
plt.savefig('../../outputs/figures/error_by_length.png', dpi=150)
plt.close()
print("Saved: error_by_length.png")

# ── 4. Confidence of errors ───────────────────────────────────────
fn = test[test['error_type'] == 'false_negative']['stress_prob']
fp = test[test['error_type'] == 'false_positive']['stress_prob']
co = test[test['error_type'] == 'correct']['stress_prob']

plt.figure(figsize=(8, 4))
plt.hist(co, bins=30, alpha=0.5, label='Correct', color='steelblue')
plt.hist(fp, bins=20, alpha=0.6, label='False Positive', color='orange')
plt.hist(fn, bins=20, alpha=0.6, label='False Negative', color='tomato')
plt.axvline(0.5, color='black', linestyle='--', linewidth=1)
plt.xlabel('Predicted Stress Probability')
plt.ylabel('Count')
plt.title('Confidence Distribution by Prediction Outcome', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('../../outputs/figures/error_confidence.png', dpi=150)
plt.close()
print("Saved: error_confidence.png")

# ── 5. Subreddit error analysis ───────────────────────────────────
sub_errors = test.groupby('subreddit').apply(
    lambda x: pd.Series({
        'total':          len(x),
        'false_negative': (x['error_type'] == 'false_negative').sum(),
        'false_positive': (x['error_type'] == 'false_positive').sum(),
        'accuracy':       (x['label'] == x['predicted']).mean()
    })
).reset_index()

print("\nError breakdown by subreddit:")
print(sub_errors.sort_values('accuracy'))

plt.figure(figsize=(10, 5))
sns.barplot(x='subreddit', y='accuracy', data=sub_errors.sort_values('accuracy'),
            palette='coolwarm')
plt.title('Model Accuracy by Subreddit (Dreaddit Test Set)', fontsize=12)
plt.xlabel('Subreddit')
plt.ylabel('Accuracy')
plt.xticks(rotation=30, ha='right')
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig('../../outputs/figures/accuracy_by_subreddit.png', dpi=150)
plt.close()
print("Saved: accuracy_by_subreddit.png")

# ── 6. False negative examples — what did we miss? ────────────────
print("\n── False Negatives (stressed posts we missed) ──")
fn_examples = test[test['error_type'] == 'false_negative'].sort_values(
    'stress_prob', ascending=True).head(5)

for _, row in fn_examples.iterrows():
    print(f"\n  Subreddit: r/{row['subreddit']} | Stress prob: {row['stress_prob']:.3f}")
    print(f"  Text: {str(row['text'])[:250]}...")

# ── 7. False positive examples ────────────────────────────────────
print("\n── False Positives (non-stressed posts flagged as stressed) ──")
fp_examples = test[test['error_type'] == 'false_positive'].sort_values(
    'stress_prob', ascending=False).head(5)

for _, row in fp_examples.iterrows():
    print(f"\n  Subreddit: r/{row['subreddit']} | Stress prob: {row['stress_prob']:.3f}")
    print(f"  Text: {str(row['text'])[:250]}...")

# ── 8. Word analysis of errors ────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

fn_text = ' '.join(test[test['error_type'] == 'false_negative']['text'].apply(clean_text))
fp_text = ' '.join(test[test['error_type'] == 'false_positive']['text'].apply(clean_text))

fn_words = Counter(fn_text.split()).most_common(15)
fp_words = Counter(fp_text.split()).most_common(15)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

words_fn, counts_fn = zip(*fn_words)
axes[0].barh(list(words_fn)[::-1], list(counts_fn)[::-1], color='tomato')
axes[0].set_title('Top Words in False Negatives\n(Stressed posts we missed)', fontsize=11)
axes[0].set_xlabel('Frequency')

words_fp, counts_fp = zip(*fp_words)
axes[1].barh(list(words_fp)[::-1], list(counts_fp)[::-1], color='orange')
axes[1].set_title('Top Words in False Positives\n(Non-stressed posts we flagged)', fontsize=11)
axes[1].set_xlabel('Frequency')

plt.tight_layout()
plt.savefig('../../outputs/figures/error_word_analysis.png', dpi=150)
plt.close()
print("Saved: error_word_analysis.png")

# ── 9. Ethics & bias summary ──────────────────────────────────────
print("\n── Ethics & Bias Summary ──")

print("\n1. CLASS PERFORMANCE GAP:")
from sklearn.metrics import f1_score
f1_stressed    = f1_score(y_test, test['predicted'], pos_label=1)
f1_notstressed = f1_score(y_test, test['predicted'], pos_label=0)
print(f"   F1 Stressed:     {f1_stressed:.4f}")
print(f"   F1 Not Stressed: {f1_notstressed:.4f}")
print(f"   Gap:             {abs(f1_stressed - f1_notstressed):.4f}")

print("\n2. SUBREDDIT BIAS:")
print("   Subreddits with lowest accuracy may indicate domain mismatch")
print("   (model trained on general stress, not domain-specific language)")
worst = sub_errors.sort_values('accuracy').head(3)
for _, row in worst.iterrows():
    print(f"   r/{row['subreddit']}: {row['accuracy']:.3f} accuracy")

print("\n3. FALSE NEGATIVE RISK (most critical for mental health tools):")
fn_rate = len(fn) / (len(fn) + (test['label'] == 1).sum())
print(f"   False negative rate: {fn_rate:.3f}")
print("   → {:.0f}% of stressed posts go undetected".format(fn_rate * 100))
print("   → In a real deployment, these would be missed students needing support")

print("\n4. KEYWORD BIAS IN STUDENT LABELS:")
print("   Soft labels were generated via keyword matching — may over-represent")
print("   explicit stress expression and under-represent implicit stress signals")

print("\n5. PRIVACY & CONSENT:")
print("   All Reddit data is public, anonymized at analysis stage")
print("   No user identification or tracking performed")
print("   Student posts should never be used for individual surveillance")

# ── 10. Save error analysis results ──────────────────────────────
test.to_csv('../../outputs/error_analysis_results.csv', index=False)
print("\n✅ Error analysis complete. All outputs saved.")
print("\nFigures saved:")
print("  error_breakdown.png")
print("  error_by_length.png")
print("  error_confidence.png")
print("  accuracy_by_subreddit.png")
print("  error_word_analysis.png")