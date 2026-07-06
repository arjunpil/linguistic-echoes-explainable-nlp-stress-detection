import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(os.path.join(OUTPUTS_DIR, 'figures'), exist_ok=True)

# ── 1. Load data & model ──────────────────────────────────────────
print("Loading data...")
with open(os.path.join(OUTPUTS_DIR, 'models', 'tfidf_vectorizer.pkl'), 'rb') as f:
    tfidf = pickle.load(f)
with open(os.path.join(OUTPUTS_DIR, 'models', 'lr_model.pkl'), 'rb') as f:
    lr = pickle.load(f)

student = pd.read_csv(os.path.join(DATA_DIR, 'reddit_student', 'student_posts_predicted.csv'))

# ── 2. Compute linguistic features per post ───────────────────────
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
        'fp_singular_rate':  sum(1 for t in tokens if t in FIRST_PERSON_SINGULAR) / n,
        'anxiety_rate':      sum(1 for t in tokens if t in ANXIETY_WORDS) / n,
        'neg_emotion_rate':  sum(1 for t in tokens if t in NEGATIVE_EMOTION) / n,
        'social_rate':       sum(1 for t in tokens if t in SOCIAL_WORDS) / n,
        'word_count':        n
    }

print("Computing linguistic features (this takes ~2 min)...")
features = student['text'].apply(compute_features)
feat_df  = pd.DataFrame(features.tolist())
student  = pd.concat([student.reset_index(drop=True), feat_df], axis=1)

# ── 3. Summary table by subreddit ─────────────────────────────────
FEATURE_COLS = ['fp_singular_rate', 'anxiety_rate',
                'neg_emotion_rate', 'social_rate', 'stress_prob']

summary = student.groupby('subreddit')[FEATURE_COLS].mean().round(4)
summary.columns = ['1st-Person Singular', 'Anxiety Vocab',
                   'Neg. Emotion', 'Social Words', 'Stress Prob']
print("\n── Subreddit Linguistic Profile ──")
print(summary.to_string())
summary.to_csv(os.path.join(OUTPUTS_DIR, 'subreddit_linguistic_profile.csv'))

# ── 4. Statistical tests: college vs others ───────────────────────
print("\n── Statistical Tests (college vs each group) ──")

college_data = student[student['subreddit'] == 'college']
comparisons  = ['GradSchool', 'PhD', 'PreMed']

results = []

for feat in FEATURE_COLS:
    for comp in comparisons:
        comp_data = student[student['subreddit'] == comp][feat]
        coll_data = college_data[feat]

        # Mann-Whitney U (non-parametric, safer for non-normal distributions)
        u_stat, p_val = stats.mannwhitneyu(coll_data, comp_data,
                                            alternative='two-sided')

        # Cohen's d effect size
        pooled_std = np.sqrt((coll_data.std()**2 + comp_data.std()**2) / 2)
        cohens_d   = (coll_data.mean() - comp_data.mean()) / (pooled_std + 1e-9)

        results.append({
            'Feature':    feat,
            'Comparison': f'college vs {comp}',
            'College Mean':  round(coll_data.mean(), 4),
            'Other Mean':    round(comp_data.mean(), 4),
            'p-value':       round(p_val, 4),
            'Cohen d':       round(cohens_d, 3),
            'Significant':   'Yes' if p_val < 0.05 else 'No'
        })

        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01
              else ('*' if p_val < 0.05 else 'ns'))
        print(f"  {feat:22s} | college vs {comp:12s} | "
              f"p={p_val:.4f}{sig:4s} | d={cohens_d:+.3f}")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUTS_DIR, 'statistical_tests.csv'), index=False)
print("\nSaved: statistical_tests.csv")

# ── 5. Main comparison figure ──────────────────────────────────────
print("\nGenerating subreddit comparison figure...")

SUBS    = ['college', 'GradSchool', 'PhD', 'PreMed']
METRICS = {
    'fp_singular_rate': 'First-Person Singular Rate',
    'anxiety_rate':     'Anxiety Vocabulary Rate',
    'neg_emotion_rate': 'Negative Emotion Rate',
    'social_rate':      'Social Words Rate',
    'stress_prob':      'Predicted Stress Probability'
}
COLORS = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a']

fig, axes = plt.subplots(1, 5, figsize=(18, 5))

for ax, (feat, label) in zip(axes, METRICS.items()):
    means = [student[student['subreddit'] == s][feat].mean() for s in SUBS]
    sems  = [student[student['subreddit'] == s][feat].sem()  for s in SUBS]

    bars = ax.bar(SUBS, means, yerr=sems, capsize=4,
                  color=COLORS, edgecolor='black', linewidth=0.6)
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Mean Rate' if feat != 'stress_prob' else 'Mean Probability')
    ax.tick_params(axis='x', rotation=25)

    # annotate significance vs college
    college_vals = student[student['subreddit'] == 'college'][feat]
    y_max = max(means) + max(sems) * 1.5
    for i, sub in enumerate(SUBS[1:], 1):
        sub_vals = student[student['subreddit'] == sub][feat]
        _, p = stats.mannwhitneyu(college_vals, sub_vals, alternative='two-sided')
        sig = '***' if p < 0.001 else ('**' if p < 0.01
              else ('*' if p < 0.05 else ''))
        if sig:
            ax.annotate(sig, xy=(i, means[i] + sems[i]),
                        ha='center', fontsize=11, color='black')

plt.suptitle('Linguistic Stress Markers Across Academic Communities\n'
             '(* p<0.05, ** p<0.01, *** p<0.001 vs r/college)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'figures', 'subreddit_linguistic_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: subreddit_linguistic_comparison.png")

# ── 6. Violin plots for richer distribution view ──────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (feat, label) in zip(axes, list(METRICS.items())[:3]):
    sns.violinplot(x='subreddit', y=feat, data=student,
                   order=SUBS, palette=COLORS, ax=ax, inner='box')
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_xlabel('Subreddit')
    ax.set_ylabel('Rate')
    ax.tick_params(axis='x', rotation=20)

plt.suptitle('Distribution of Key Stress Markers by Community', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'figures', 'subreddit_violin.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: subreddit_violin.png")

# ── 7. Effect size heatmap ────────────────────────────────────────
pivot = results_df.pivot(index='Feature', columns='Comparison', values='Cohen d')

plt.figure(figsize=(9, 5))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, linewidths=0.5,
            cbar_kws={'label': "Cohen's d (positive = college higher)"})
plt.title("Effect Sizes (Cohen's d): r/college vs Other Communities", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'figures', 'effect_size_heatmap.png'), dpi=150)
plt.close()
print("Saved: effect_size_heatmap.png")

print("\n✅ Subreddit analysis complete.")
print("\nKey figures saved:")
print("  subreddit_linguistic_comparison.png  ← main figure for paper")
print("  subreddit_violin.png")
print("  effect_size_heatmap.png")