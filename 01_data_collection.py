import urllib.request
import zipfile
import os
import pandas as pd
import json

# ── 1. Download Dreaddit ──────────────────────────────────────────
os.makedirs('../data/dreaddit', exist_ok=True)

print("Downloading Dreaddit...")
urllib.request.urlretrieve(
    'http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip',
    '../data/dreaddit/dreaddit.zip'
)

with zipfile.ZipFile('../data/dreaddit/dreaddit.zip', 'r') as z:
    z.extractall('../data/dreaddit/')

print("Done. Files:", os.listdir('../data/dreaddit/'))

# ── 2. Inspect Dreaddit ───────────────────────────────────────────
train = pd.read_csv('../data/dreaddit/dreaddit-train.csv')
test  = pd.read_csv('../data/dreaddit/dreaddit-test.csv')

print("\nTrain shape:", train.shape)
print("Test shape: ", test.shape)
print("\nColumns:", train.columns.tolist())
print("\nLabel distribution (train):")
print(train['label'].value_counts())
print("\nSubreddits in dataset:")
print(train['subreddit'].value_counts())
print("\nSample text (first 300 chars):")
print(train['text'].iloc[0][:300])

# ── 3. Load & sample Reddit student posts ─────────────────────────
os.makedirs('../data/reddit_student', exist_ok=True)

FILES = {
    'college':    '../data/reddit_student/r_college_posts.jsonl',
    'GradSchool': '../data/reddit_student/r_GradSchool_posts.jsonl',
    'PhD':        '../data/reddit_student/r_PhD_posts.jsonl',
    'PreMed':     '../data/reddit_student/r_PreMed_posts.jsonl',
}

SAMPLE_PER_SUB = 2000

all_posts = []

for sub, filepath in FILES.items():
    print(f"\nLoading r/{sub}...")
    posts = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                post = json.loads(line)
                text = post.get('selftext', '') or ''
                title = post.get('title', '') or ''

                if (len(text) > 100
                        and text not in ('[removed]', '[deleted]')
                        and title not in ('[removed]', '[deleted]')):
                    posts.append({
                        'subreddit':    sub,
                        'title':        title,
                        'text':         text,
                        'score':        post.get('score', 0),
                        'num_comments': post.get('num_comments', 0),
                        'created_utc':  post.get('created_utc', ''),
                        'id':           post.get('id', '')
                    })
            except json.JSONDecodeError:
                continue

    df_sub = pd.DataFrame(posts)
    if len(df_sub) > SAMPLE_PER_SUB:
        df_sub = df_sub.sample(n=SAMPLE_PER_SUB, random_state=42)

    print(f"  {len(df_sub)} usable posts from r/{sub}")
    all_posts.append(df_sub)

# ── 4. Save combined student dataset ─────────────────────────────
df_student = pd.concat(all_posts, ignore_index=True).drop_duplicates(subset='id')
df_student.to_csv('../data/reddit_student/student_posts_raw.csv', index=False)

print(f"\nTotal student posts saved: {len(df_student)}")
print(df_student['subreddit'].value_counts())
print("\nSample text:")
print(df_student['text'].iloc[0][:300])