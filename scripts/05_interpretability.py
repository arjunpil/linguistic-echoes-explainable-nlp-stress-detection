import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import shap
import lime
import lime.lime_text
from sklearn.pipeline import make_pipeline
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('../../outputs/figures', exist_ok=True)

# ── 1. Load model & vectorizer ────────────────────────────────────
print("Loading model and vectorizer...")
with open('../../outputs/models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('../../outputs/models/lr_model.pkl', 'rb') as f:
    lr = pickle.load(f)

# ── 2. Load data ──────────────────────────────────────────────────
print("Loading data...")
test     = pd.read_csv('../../data/dreaddit/dreaddit-test.csv')
student  = pd.read_csv('../../data/reddit_student/student_posts_predicted.csv')

X_test  = test['text'].astype(str)
y_test  = test['label']

# ── 3. SHAP — global feature importance ───────────────────────────
print("\nRunning SHAP...")
X_test_tfidf = tfidf.transform(X_test)

# Use linear explainer for LR — fast and exact
explainer   = shap.LinearExplainer(lr, X_test_tfidf, feature_perturbation='interventional')
shap_values = explainer.shap_values(X_test_tfidf)

feature_names = tfidf.get_feature_names_out()

# Global mean absolute SHAP values
mean_shap = np.abs(shap_values).mean(axis=0)
top_n     = 20
top_idx   = np.argsort(mean_shap)[-top_n:][::-1]

top_features = [feature_names[i] for i in top_idx]
top_shap     = [mean_shap[i] for i in top_idx]

plt.figure(figsize=(10, 6))
colors = ['tomato' if lr.coef_[0][top_idx[i]] > 0 else 'steelblue'
          for i in range(top_n)]
plt.barh(top_features[::-1], top_shap[::-1], color=colors[::-1])
plt.xlabel('Mean |SHAP Value|')
plt.title('Top 20 Features by SHAP Importance\n(Red = predicts Stressed, Blue = predicts Not Stressed)',
          fontsize=12)
plt.tight_layout()
plt.savefig('../../outputs/figures/shap_global.png', dpi=150)
plt.close()
print("Saved: shap_global.png")

# ── 4. SHAP — per-class breakdown ────────────────────────────────
# Separate SHAP values for stressed vs not stressed examples
stressed_idx     = np.where(y_test.values == 1)[0]
notstressed_idx  = np.where(y_test.values == 0)[0]

shap_stressed    = shap_values[stressed_idx].mean(axis=0)
shap_notstressed = shap_values[notstressed_idx].mean(axis=0)

# Top pushing toward stressed
top_push_stressed = np.argsort(shap_stressed)[-15:][::-1]
# Top pushing toward not stressed  
top_push_notstressed = np.argsort(shap_notstressed)[:15]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].barh(
    [feature_names[i] for i in top_push_stressed][::-1],
    [shap_stressed[i] for i in top_push_stressed][::-1],
    color='tomato'
)
axes[0].set_title('Features Driving STRESSED Prediction', fontsize=11)
axes[0].set_xlabel('Mean SHAP Value')

axes[1].barh(
    [feature_names[i] for i in top_push_notstressed][::-1],
    [shap_notstressed[i] for i in top_push_notstressed][::-1],
    color='steelblue'
)
axes[1].set_title('Features Driving NOT STRESSED Prediction', fontsize=11)
axes[1].set_xlabel('Mean SHAP Value')

plt.suptitle('SHAP Values by Class', fontsize=13)
plt.tight_layout()
plt.savefig('../../outputs/figures/shap_by_class.png', dpi=150)
plt.close()
print("Saved: shap_by_class.png")

# ── 5. LIME — individual explanations ────────────────────────────
print("\nRunning LIME on individual examples...")

# Build pipeline for LIME (needs raw text → probability)
pipeline = make_pipeline(tfidf, lr)

def predictor(texts):
    return pipeline.predict_proba(texts)

lime_explainer = lime.lime_text.LimeTextExplainer(
    class_names=['Not Stressed', 'Stressed'],
    random_state=42
)

# Pick 3 stressed + 3 not-stressed examples from test set
stressed_examples    = test[test['label'] == 1]['text'].iloc[:3].tolist()
notstressed_examples = test[test['label'] == 0]['text'].iloc[:3].tolist()

examples = (
    [(txt, 'Stressed')    for txt in stressed_examples] +
    [(txt, 'Not Stressed') for txt in notstressed_examples]
)

lime_results = []

for i, (text, true_label) in enumerate(examples):
    exp = lime_explainer.explain_instance(
        text,
        predictor,
        num_features=10,
        num_samples=500
    )
    lime_results.append((text, true_label, exp))

    # Save individual LIME figure
    fig = exp.as_pyplot_figure(label=1)
    fig.suptitle(f'LIME — Example {i+1} (True: {true_label})', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'../../outputs/figures/lime_example_{i+1}.png', dpi=150)
    plt.close()
    print(f"  Saved: lime_example_{i+1}.png (True label: {true_label})")

# ── 6. LIME summary — top words across examples ───────────────────
print("\nLIME word importance summary:")
all_lime_words = {}

for text, true_label, exp in lime_results:
    for word, weight in exp.as_list(label=1):
        if word not in all_lime_words:
            all_lime_words[word] = []
        all_lime_words[word].append(weight)

avg_lime = {w: np.mean(v) for w, v in all_lime_words.items()}
sorted_lime = sorted(avg_lime.items(), key=lambda x: abs(x[1]), reverse=True)[:20]

print("\nTop LIME words (averaged across examples):")
for word, weight in sorted_lime:
    direction = 'STRESSED' if weight > 0 else 'NOT STRESSED'
    print(f"  {word:25s} {weight:+.4f}  → {direction}")

words_l  = [w for w, _ in sorted_lime]
weights_l = [v for _, v in sorted_lime]
colors_l  = ['tomato' if v > 0 else 'steelblue' for v in weights_l]

plt.figure(figsize=(10, 6))
plt.barh(words_l[::-1], weights_l[::-1], color=colors_l[::-1])
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('Mean LIME Weight')
plt.title('Top LIME Features Averaged Across Examples\n(Red = Stressed, Blue = Not Stressed)',
          fontsize=12)
plt.tight_layout()
plt.savefig('../../outputs/figures/lime_summary.png', dpi=150)
plt.close()
print("Saved: lime_summary.png")

# ── 7. Stress probability distribution — student data ─────────────
print("\nPlotting student stress probability distribution...")
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution by subreddit
for sub in student['subreddit'].unique():
    subset = student[student['subreddit'] == sub]['stress_prob']
    axes[0].hist(subset, bins=30, alpha=0.5, label=f'r/{sub}')

axes[0].set_xlabel('Stress Probability')
axes[0].set_ylabel('Count')
axes[0].set_title('Stress Probability Distribution by Subreddit')
axes[0].legend()

# Boxplot
sns.boxplot(x='subreddit', y='stress_prob', data=student,
            palette='Set2', ax=axes[1])
axes[1].set_title('Stress Probability by Subreddit (Boxplot)')
axes[1].set_xlabel('Subreddit')
axes[1].set_ylabel('Predicted Stress Probability')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('../../outputs/figures/student_stress_distribution.png', dpi=150)
plt.close()
print("Saved: student_stress_distribution.png")

# ── 8. High stress student examples ──────────────────────────────
print("\n── High Stress Student Posts (prob > 0.85) ──")
high_stress = student[student['stress_prob'] > 0.85].sort_values(
    'stress_prob', ascending=False)

print(f"Total high-stress posts: {len(high_stress)}")
print(f"By subreddit:\n{high_stress['subreddit'].value_counts()}")
print("\nTop 5 highest stress probability posts:")

for _, row in high_stress.head(5).iterrows():
    print(f"\n  Subreddit: r/{row['subreddit']} | Prob: {row['stress_prob']:.3f}")
    print(f"  Text: {str(row['text'])[:200]}...")

high_stress.to_csv('../../outputs/high_stress_student_posts.csv', index=False)
print("\nSaved: high_stress_student_posts.csv")

print("\n✅ Interpretability analysis complete.")