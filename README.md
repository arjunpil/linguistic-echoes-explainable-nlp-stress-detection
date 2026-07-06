# Linguistic Echoes: Explainable NLP for Academic Stress Detection

**Project Track:** Humanities & Social Science (HSS)  
**Competition:** IAI²O AI for Science (AI4Sci) 2026  
**Author:** Arjun Pillai  
**School:** Irvington High School, USA

---

## Overview

This repository contains the code for the project **"Linguistic Echoes: Explainable NLP for Academic Stress Detection"**.

The project explores whether modern NLP models can recover established psycholinguistic stress markers and whether different academic communities (undergrad, grad, PhD, pre-med) express stress differently.

**Key Highlights:**
- Fine-tuned DistilBERT + TF-IDF baseline on Dreaddit dataset
- Multi-method explainability (SHAP, LIME, LIWC)
- Analysis of 8,000 academic Reddit posts
- Statistical community comparison with Benjamini-Hochberg correction

---

## Repository Structure

linguistic-echoes-explainable-nlp-stress-detection/
├── data/                    # (Not stored here - see Dataset section)
├── scripts/               # Main analysis notebooks
│   ├── 01_data_collection.py
│   ├── 02_eda.py
│   ├── 03_baseline_model.py
|   ├── 04_transformer_model.py           # Ran on Kaggle
|   ├── 05_interpretability.py
|   ├── 06_error_analysis.py
|   ├── 07_subreddit_analysis.py
│   └── 08_additional_analysis.py
├── outputs/                     # Python scripts
│   ├── figures/
    ├── models/
    ├── error_analysis_results.csv
    ├── high_stress_student_posts.csv
    ├── statistical_tests.csv
    ├── subreddit_linguistic_profile.csv
├── reports/
    ├── HSS_TeamMechInterp_LinguisticEchoesExplainableNLPforAcademicStressDetection_Paper.pdf
    ├── HSS_TeamMechInterp_LinguisticEchoesExplainableNLPforAcademicStressDetection_QuadChart.pdf
    ├── hss_linguisticechoes_quadchart.html                      #HTML to generate PDF
├── requirements.txt
├── .gitignore
└── README.md

---

##Dataset

The full Dataset can be found here: https://huggingface.co/datasets/arjunpil/linguistic-echoes-explainable-nlp-stress-detection
It contains the Dreaddit and the Reddit Student responses from many different subreddits.
You can download it and adjust the file pathing within the file per your circumstances.

---

##Technologies Used

Models: DistilBERT (Hugging Face), TF-IDF + Logistic Regression
Explainability: SHAP, LIME, LIWC
Statistics: SciPy, statsmodels, Pingouin
Visualization: Matplotlib, Seaborn, WordCloud
Others: pandas, numpy, scikit-learn

---

##Results

Best Model: DistilBERT → 79% Accuracy, 0.874 ROC-AUC
Strong convergence between SHAP, LIME, and LIWC on first-person singular pronouns and negative emotion
Significant linguistic differences across academic communities (p < 0.001 after correction)
Full paper and quad chart available in the reports/ folder.
---

##License

This project is licensed under the MIT License — see LICENSE file.

---
