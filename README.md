# Census Income Classification

An end-to-end binary classification pipeline predicting whether an individual's income is above or below $50K, using the UCI Adult/Census Income dataset.

## Overview

This project walks through the full machine learning lifecycle — data cleaning, exploratory analysis, feature engineering, model training, and evaluation — culminating in a comparison between a traditional ML model and a neural network on the same feature set.

## Key Steps

**Data Cleaning**
- Identified missing values across five columns (`age`, `workclass`, `occupation`, `hours-per-week`, `native-country`)
- Dropped rows with missing `age`/`native-country` (small % of ~32,500 total rows)
- Found that missingness in `workclass` and `occupation` was correlated, likely reflecting non-working individuals — imputed as `"Unknown"` rather than dropping or using mode
- Imputed `hours-per-week` with the median after checking its distribution

**Exploratory Data Analysis**
- Identified a ~75/25 class imbalance (`<=50K` majority) and prioritized F1 score over raw accuracy as the evaluation metric as a result
- Used pairplots and count plots to assess feature-label separation; found `age`, `education-num`, `capital-gain`, and `capital-loss` showed strong class separation
- Noted that `capital-gain`/`capital-loss` are zero-inflated, which can mislead standard visualizations if not accounted for
- Flagged `education` and `education-num` as likely redundant features

**Modeling**
- Trained and tuned a logistic regression model: **85% accuracy, 0.65 F1**
- Built and trained a neural network (Keras) on the same feature set for comparison
- Ultimately recommended logistic regression, given comparable performance and stronger interpretability for this use case

## Tech Stack
Python, pandas, scikit-learn, Keras, Matplotlib/Seaborn

## Files
- `Capstone.ipynb` — full pipeline and analysis

---
*Completed as the capstone project for Break Through Tech's Machine Learning Foundations fellowship, in partnership with Cornell Tech.*
