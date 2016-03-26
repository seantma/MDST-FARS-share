# MDST-FARS-share

This repo is a start where Tiapei Xie, Sheng Yang and Sean Ma are collaborating together for the goal of becoming the top notch Data Scientiest. We started with the the MDST-FARS competition.

## Mission:
### Helping each other to become the top Data Scientist in our field.

## Goals:
We use competitions in Kaggle to sharpen our skill sets. Our main goal is to get placed in the public leader board. Pratical goals for now are:

1. Further test FARS data set with ensemble technologies (Tianpei & Sean).
2. Practice deep learning on FARS data set (Tianpei lead).
3. Continue to learn from 2 public Kaggle competitions (Home Depot; Santander) (Sheng lead).
5. Learn cloud technologies such as AWS (Sean lead).
4. Compete in a public Kaggle competition (maybe in May or June).

------

## Models and AUC scores:

| Model code                           | Submission CSV                                     | Public Score | Private Score |Note   |
|--------------------------------------|----------------------------------------------------|--------------|---------------|-------|
| Model_xgboost-production_weighted.py |fars_submit_xgb004-production_weighted_missing.csv | 0.87135      |	0.86657    |No change using missing option |
| Model_xgboost-production_weighted.py |fars_submit_xgb004-production_weighted.csv| 0.87135      |	0.86657    |       |
| Model_xgboost-production_weighted_Tian_acc_per.py|fars_submit_xgb005-production_weighted_Tian_acc_per.csv | 0.85757    |	0.85050    | Seems like 2 tables with dummies are ok    |
  