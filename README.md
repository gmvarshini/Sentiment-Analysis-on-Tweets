# Sentiment Analysis on Tweets

Classifies tweets as hateful (racist or sexist) or not, using classic NLP
preprocessing and machine learning models.

## Dataset

- `train.csv`: 31,962 labelled tweets with the columns `id`, `label` and
  `tweet`. Label 1 means the tweet is racist or sexist. Only about 7% of tweets
  (2,242) have label 1, so the data is imbalanced.
- `test.csv`: 17,197 unlabelled tweets used for final predictions.

## Approach

1. **Text cleaning.** Lowercase, remove punctuation and numbers, remove English
   stopwords (NLTK), and lemmatize each word (TextBlob).
2. **Train/test split.** 80% for training, 20% held out for evaluation.
3. **Vectorizing.** Two ways of turning text into numbers: word counts
   (`CountVectorizer`) and TF-IDF (`TfidfVectorizer`).
4. **Models.** Logistic Regression, XGBoost and LightGBM, each trained with both
   vectorizers, so six combinations in total.
5. **Evaluation.** Cross validated accuracy on the held out data, and a ROC
   curve.
6. **Prediction.** Labels for the unlabelled test tweets.

## Results

| Model | Count vectors | TF-IDF |
| --- | --- | --- |
| Logistic Regression | **94.8%** | 93.1% |
| XGBoost | 94.3% | 94.0% |
| LightGBM | 93.7% | 93.7% |

Logistic Regression with simple word counts performed best.

## What I would do differently today

- **Look beyond accuracy.** Because only 7% of tweets are hateful, a model that
  always predicts "not hateful" already gets about 93% accuracy. Precision,
  recall and F1 for the hateful class would show the real quality.
- **Handle the imbalance.** Use class weights or resampling so the model pays
  attention to the rare class.
- **Evaluate the trained models directly.** The notebook re-trains models inside
  cross validation on the held out set. Scoring the already trained models on
  that set would be cleaner.
- **Try sentence embeddings or a fine tuned transformer.** These usually
  understand context, sarcasm and slang better than word counts.

## How to run

Open `Sentiment_Analysis_on_Tweets.ipynb` in Jupyter or Google Colab and run all
cells. It needs pandas, NumPy, scikit-learn, NLTK, TextBlob, XGBoost, LightGBM,
Matplotlib and WordCloud.
