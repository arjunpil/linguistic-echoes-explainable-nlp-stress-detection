import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(os.path.join(OUTPUTS_DIR, 'figures'), exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────
print("Loading data...")
student = pd.read_csv(os.path.join(DATA_DIR, 'reddit_student', 'student_posts_predicted.csv'))

# recompute linguistic features (same as script 07)
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

FIRST_PERSON_SINGULAR = {'i', 'me', 'my', 'myself', 'mine', 'im', "i'm",
                          "i've", "i'll", "i'd"}
ANXIETY_WORDS         = {'anxious', 'anxiety', 'worried', 'worry', 'nervous',
                          'scared', 'fear', 'panic', 'overwhelmed', 'stressed',
                          'stress', 'dread', 'afraid', 'terrified', 'tense'}
NEGATIVE_EMOTION      = {'sad', 'depressed', 'hopeless', 'worthless', 'miserable',
                          'exhausted', 'burnout', 'hate', 'angry', 'frustrated',
                          'upset', 'devastated', 'broken', 'failed', 'failing',
                          'cant', "can't", 'helpless', 'lost'}
SOCIAL_WORDS          = {'friend', 'friends', 'family', 'people', 'everyone',
                          'someone', 'together', 'community', 'support', 'help',
                          'you', 'we', 'us', 'our', 'your', 'they', 'them',
                          'colleague', 'advisor', 'professor'}

def compute_features(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    tokens = word_tokenize(re.sub(r'[^a-z\s]', '', text))
    n = max(len(tokens), 1)
    return {
        'fp_singular_rate': sum(1 for t in tokens if t in FIRST_PERSON_SINGULAR) / n,
        'anxiety_rate':     sum(1 for t in tokens if t in ANXIETY_WORDS) / n,
        'neg_emotion_rate': sum(1 for t in tokens if t in NEGATIVE_EMOTION) / n,
        'social_rate':      sum(1 for t in tokens if t in SOCIAL_WORDS) / n,
    }

print("Computing features (~2 min)...")
features = student['text'].apply(compute_features)
feat_df  = pd.DataFrame(features.tolist())
student  = pd.concat([student.reset_index(drop=True), feat_df], axis=1)

# ── 2. Correlation heatmap ────────────────────────────────────────
print("Generating correlation heatmap...")

corr_cols = ['stress_prob', 'fp_singular_rate', 'anxiety_rate',
             'neg_emotion_rate', 'social_rate']
corr_labels = ['Stress\nProbability', 'First-Person\nSingular',
               'Anxiety\nVocab', 'Negative\nEmotion', 'Social\nWords']

corr_matrix = student[corr_cols].corr(method='spearman')

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.3f',
    cmap='RdBu_r',
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    xticklabels=corr_labels,
    yticklabels=corr_labels,
    ax=ax,
    cbar_kws={'label': "Spearman's ρ"}
)
ax.set_title("Spearman Correlations: Stress Probability vs Linguistic Features\n"
             "(Student Reddit Corpus, n=8,000)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'figures', 'correlation_heatmap.png'), dpi=150)
plt.close()
print("Saved: correlation_heatmap.png")

# Print key correlations
print("\nKey Spearman correlations with stress_prob:")
for col, label in zip(corr_cols[1:], corr_labels[1:]):
    r, p = stats.spearmanr(student['stress_prob'], student[col])
    print(f"  {label.replace(chr(10), ' '):25s} ρ={r:+.3f}  p={p:.4e}")

# ── 3. Benjamini-Hochberg correction ─────────────────────────────
print("\nApplying Benjamini-Hochberg correction...")

# Reload raw p-values from script 07
results_df = pd.read_csv(os.path.join(OUTPUTS_DIR, 'statistical_tests.csv'))

p_values  = results_df['p-value'].values
features  = results_df['Feature'].values
comps     = results_df['Comparison'].values
cohens_d  = results_df['Cohen d'].values

reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

results_df['p_corrected_BH'] = p_corrected.round(4)
results_df['Significant_BH'] = reject

print("\n── Results after Benjamini-Hochberg FDR correction (α=0.05) ──")
print(f"{'Feature':<25} {'Comparison':<25} {'p-raw':>8} {'p-BH':>8} "
      f"{'d':>7} {'Sig?':>6}")
print("-" * 85)
for _, row in results_df.iterrows():
    sig = '✓' if row['Significant_BH'] else '✗'
    print(f"{row['Feature']:<25} {row['Comparison']:<25} "
          f"{row['p-value']:>8.4f} {row['p_corrected_BH']:>8.4f} "
          f"{row['Cohen d']:>7.3f} {sig:>6}")

n_sig_before = (results_df['p-value'] < 0.05).sum()
n_sig_after  = results_df['Significant_BH'].sum()
print(f"\nSignificant before correction: {n_sig_before}/{len(results_df)}")
print(f"Significant after BH correction: {n_sig_after}/{len(results_df)}")

results_df.to_csv(os.path.join(OUTPUTS_DIR, 'statistical_tests_corrected.csv'), index=False)
print("Saved: statistical_tests_corrected.csv")

# ── 4. False negative / positive examples ─────────────────────────
print("\n── Concrete Error Examples ──")

test = pd.read_csv(os.path.join(DATA_DIR, 'dreaddit', 'dreaddit-test.csv'))
with open(os.path.join(OUTPUTS_DIR, 'models', 'tfidf_vectorizer.pkl'), 'rb') as f:
    tfidf = pickle.load(f)
with open(os.path.join(OUTPUTS_DIR, 'models', 'lr_model.pkl'), 'rb') as f:
    lr = pickle.load(f)

X_test_tfidf       = tfidf.transform(test['text'].astype(str))
test['predicted']  = lr.predict(X_test_tfidf)
test['stress_prob'] = lr.predict_proba(X_test_tfidf)[:, 1]
test['error_type'] = 'correct'
test.loc[(test['label']==1) & (test['predicted']==0), 'error_type'] = 'false_negative'
test.loc[(test['label']==0) & (test['predicted']==1), 'error_type'] = 'false_positive'

# Pick clearest examples — FN with very low stress prob, FP with very high
fn_examples = test[test['error_type']=='false_negative'].sort_values(
    'stress_prob').head(3)
fp_examples = test[test['error_type']=='false_positive'].sort_values(
    'stress_prob', ascending=False).head(3)

print("\nFALSE NEGATIVES (stressed posts predicted as not stressed):")
for i, (_, row) in enumerate(fn_examples.iterrows(), 1):
    print(f"\n  Example {i} | r/{row['subreddit']} | "
          f"Stress prob: {row['stress_prob']:.3f}")
    print(f"  \"{str(row['text'])[:300]}\"")

print("\nFALSE POSITIVES (not-stressed posts predicted as stressed):")
for i, (_, row) in enumerate(fp_examples.iterrows(), 1):
    print(f"\n  Example {i} | r/{row['subreddit']} | "
          f"Stress prob: {row['stress_prob']:.3f}")
    print(f"  \"{str(row['text'])[:300]}\"")

print("\n✅ Additional analysis complete.")
print("Figures: correlation_heatmap.png")
print("Tables:  statistical_tests_corrected.csv")