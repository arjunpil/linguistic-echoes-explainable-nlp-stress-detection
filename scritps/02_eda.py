import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re
import os

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

os.makedirs('../outputs/figures', exist_ok=True)

STOP_WORDS = set(stopwords.words('english'))

# ── 1. Load data ──────────────────────────────────────────────────
print("Loading data...")
dread_train = pd.read_csv('../data/dreaddit/dreaddit-train.csv')
dread_test  = pd.read_csv('../data/dreaddit/dreaddit-test.csv')
dread       = pd.concat([dread_train, dread_test], ignore_index=True)

student     = pd.read_csv('../data/reddit_student/student_posts_raw.csv')

print(f"Dreaddit: {len(dread)} posts")
print(f"Student:  {len(student)} posts")

# ── 2. Soft-label student posts ───────────────────────────────────
STRESS_KEYWORDS = [
    'overwhelmed', 'burnout', 'burnt out', 'burned out', 'exhausted',
    'anxious', 'anxiety', 'panic', 'depressed', 'hopeless', 'struggling',
    'stressed', 'breaking down', 'can\'t cope', 'falling behind',
    'want to quit', 'give up', 'no motivation', 'can\'t sleep',
    'too much', 'drowning', 'failing', 'lost', 'worthless', 'helpless'
]

def soft_label(text):
    text_lower = str(text).lower()
    hits = sum(1 for kw in STRESS_KEYWORDS if kw in text_lower)
    if hits >= 2:
        return 1   # stressed
    elif hits == 0:
        return 0   # not stressed
    else:
        return -1  # ambiguous

student['label'] = student['text'].apply(soft_label)

print("\nStudent soft label distribution:")
print(student['label'].value_counts())

# drop ambiguous for training use
student_labeled = student[student['label'] != -1].copy()
print(f"Usable after removing ambiguous: {len(student_labeled)}")

student_labeled.to_csv('../data/reddit_student/student_posts_labeled.csv', index=False)

# ── 3. Text cleaning ──────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)       # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)      # keep only letters
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

print("\nCleaning text...")
dread['clean_text']           = dread['text'].apply(clean_text)
student_labeled['clean_text'] = student_labeled['text'].apply(clean_text)

# ── 4. Class distribution — Dreaddit ─────────────────────────────
plt.figure(figsize=(6, 4))
counts = dread['label'].value_counts().sort_index()
sns.barplot(x=['Not Stressed (0)', 'Stressed (1)'], y=counts.values, palette='coolwarm')
plt.title('Dreaddit — Class Distribution', fontsize=13)
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('../outputs/figures/dreaddit_class_dist.png', dpi=150)
plt.close()
print("Saved: dreaddit_class_dist.png")

# ── 5. Class distribution — Student posts ────────────────────────
plt.figure(figsize=(6, 4))
scounts = student_labeled['label'].value_counts().sort_index()
sns.barplot(x=['Not Stressed (0)', 'Stressed (1)'], y=scounts.values, palette='coolwarm')
plt.title('Student Reddit — Soft Label Distribution', fontsize=13)
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('../outputs/figures/student_class_dist.png', dpi=150)
plt.close()
print("Saved: student_class_dist.png")

# ── 6. Text length distribution ───────────────────────────────────
dread['word_count']           = dread['text'].apply(lambda x: len(str(x).split()))
student_labeled['word_count'] = student_labeled['text'].apply(lambda x: len(str(x).split()))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.boxplot(x='label', y='word_count', data=dread, palette='coolwarm', ax=axes[0])
axes[0].set_title('Dreaddit — Word Count by Label')
axes[0].set_xlabel('Label (0=Not Stressed, 1=Stressed)')
axes[0].set_ylabel('Word Count')
axes[0].set_ylim(0, 500)

sns.boxplot(x='subreddit', y='word_count', data=student_labeled, palette='Set2', ax=axes[1])
axes[1].set_title('Student Reddit — Word Count by Subreddit')
axes[1].set_xlabel('Subreddit')
axes[1].set_ylabel('Word Count')
axes[1].set_ylim(0, 800)
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('../outputs/figures/text_length_dist.png', dpi=150)
plt.close()
print("Saved: text_length_dist.png")

# ── 7. Word clouds — Dreaddit by label ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, label, title in zip(axes, [0, 1], ['Not Stressed', 'Stressed']):
    corpus = ' '.join(dread[dread['label'] == label]['clean_text'].tolist())
    wc = WordCloud(width=800, height=400, background_color='white',
                   colormap='RdYlGn_r' if label == 1 else 'Blues',
                   max_words=80).generate(corpus)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Dreaddit — {title}', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/figures/wordcloud_dreaddit.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: wordcloud_dreaddit.png")

# ── 8. Word clouds — Student posts by label ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, label, title in zip(axes, [0, 1], ['Not Stressed', 'Stressed']):
    subset = student_labeled[student_labeled['label'] == label]
    corpus = ' '.join(subset['clean_text'].tolist())
    wc = WordCloud(width=800, height=400, background_color='white',
                   colormap='RdYlGn_r' if label == 1 else 'Blues',
                   max_words=80).generate(corpus)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Student Reddit — {title}', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/figures/wordcloud_student.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: wordcloud_student.png")

# ── 9. Top 20 words per label — Dreaddit ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, label, title in zip(axes, [0, 1], ['Not Stressed', 'Stressed']):
    corpus = ' '.join(dread[dread['label'] == label]['clean_text'].tolist()).split()
    top = Counter(corpus).most_common(20)
    words, counts = zip(*top)
    sns.barplot(x=list(counts), y=list(words), palette='coolwarm', ax=ax)
    ax.set_title(f'Dreaddit Top Words — {title}', fontsize=12)
    ax.set_xlabel('Frequency')

plt.tight_layout()
plt.savefig('../outputs/figures/top_words_dreaddit.png', dpi=150)
plt.close()
print("Saved: top_words_dreaddit.png")

# ── 10. LIWC feature comparison ───────────────────────────────────
liwc_features = [
    'lex_liwc_anx', 'lex_liwc_negemo', 'lex_liwc_sad',
    'lex_liwc_anger', 'lex_liwc_posemo', 'lex_liwc_cogproc',
    'lex_liwc_focuspast', 'lex_liwc_focusfuture', 'lex_liwc_i'
]

liwc_means = dread.groupby('label')[liwc_features].mean().T
liwc_means.columns = ['Not Stressed', 'Stressed']

plt.figure(figsize=(10, 5))
liwc_means.plot(kind='bar', color=['steelblue', 'tomato'], figsize=(10, 5))
plt.title('LIWC Feature Means by Stress Label (Dreaddit)', fontsize=13)
plt.xlabel('LIWC Feature')
plt.ylabel('Mean Score')
plt.xticks(rotation=30, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('../outputs/figures/liwc_comparison.png', dpi=150)
plt.close()
print("Saved: liwc_comparison.png")

print("\n✅ EDA complete. All figures saved to outputs/figures/")
print("\nKey stats:")
print(f"  Dreaddit stressed:     {(dread['label']==1).sum()}")
print(f"  Dreaddit not stressed: {(dread['label']==0).sum()}")
print(f"  Student stressed:      {(student_labeled['label']==1).sum()}")
print(f"  Student not stressed:  {(student_labeled['label']==0).sum()}")
print(f"  Student ambiguous:     {(student['label']==-1).sum()}")