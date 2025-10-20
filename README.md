# Natural Language Processing(NLP):  Sentiment Analysis with Hugging Face Transformers

A Natural Language Processing (NLP) project that uses **Hugging Face Transformers** to perform **sentiment analysis** on text data.  
The goal is to automatically classify text as **Positive**, **Negative**, or **Neutral**, evaluate multiple models, visualize results, and apply sentiment detection to a creative domain of our choice.


## 🧠 Project Overview

The project demonstrates how pretrained Transformer models can be used for **sentiment classification**, **evaluation**, and **domain adaptation**.

### Objectives

1. Load and use a pretrained sentiment model with the Hugging Face `pipeline`.
2. Run predictions on tweets and labeled text data (e.g., SMILE Twitter Emotion dataset).
3. Evaluate model performance using standard metrics.
4. Visualize sentiment distributions and model confidence.
5. Compare two Hugging Face sentiment models.
6. Apply sentiment analysis creatively (e.g., news articles, song lyrics, or YouTube comments).
7. Fine-tune an existing model for domain-specific data.


## ⚙️ Models Used

| Model Name                                        | Description                                      | Label Scheme                  |
| ------------------------------------------------- | ------------------------------------------------ | ----------------------------- |
| `distilbert-base-uncased-finetuned-sst-2-english` | Compact BERT variant fine-tuned on SST-2 dataset | POSITIVE / NEGATIVE           |
| `cardiffnlp/twitter-roberta-base-sentiment`       | RoBERTa model trained on 60 M tweets             | NEGATIVE / NEUTRAL / POSITIVE |


## 📂 Dataset

**Dataset:** [SMILE Twitter Emotion Dataset (Kaggle)](https://www.kaggle.com/datasets/ashkhagan/smile-twitter-emotion-dataset)  
**Description:** Short tweets labeled by emotion.  
We map emotion labels → sentiment categories (POSITIVE / NEGATIVE / NEUTRAL).

## 📈 Workflow

### 🔹 1. Data Preparation

- Load and inspect the SMILE dataset (`smile-annotations-final.csv`)
- Clean text: lowercase, remove URLs, mentions, hashtags
- Map emotion labels to broader sentiment categories

### 🔹 2. Modeling

- Build two Hugging Face pipelines for sentiment analysis
- Normalize label outputs across models
- Batch predictions for efficiency

### 🔹 3. Evaluation

- Metrics: **Accuracy**, **Macro F1**, and **Confusion Matrix**
- Visualize:
  - Sentiment distribution
  - Model confidence histogram

### 🔹 4. Comparison

- Compare DistilBERT vs Twitter RoBERTa
- Analyze which performs better on short informal text

### 🔹 5. Creative Application

- Apply best model to a custom dataset (lyrics, news, comments)
- Analyze trends and visualize sentiment breakdown

### 🔹 6. Fine-Tuning

- Optional fine-tuning script using Hugging Face `Trainer`
- Adapt model for new domain (e.g., lyrics dataset)


## 🧩 Evaluation Example

| Model              | Accuracy | Macro-F1 |
| ------------------ | -------- | -------- |
| DistilBERT (SST-2) | 0.85     | 0.83     |
| Twitter RoBERTa    | 0.88     | 0.86     |

**Insight:**  
RoBERTa tends to perform better on social-media-style text due to domain pretraining.


## 📊 Visualizations

- **Sentiment distribution bar chart**
- **Prediction confidence histogram**
- **Confusion matrix heatmap**


## 🧮 Technologies

- Python 3
- pandas, NumPy, scikit-learn
- matplotlib
- transformers (Hugging Face)
- torch

## 📁 Project Structure
```bash
Sentiment-analysis-using-Hugging-Face-Transformer/
├── data/
│ └── smile-annotations-final.csv
├── notebooks/
│ └── Sentiment_Analysis.ipynb
├── outputs/
│ ├── confusion_matrices/
│ └── sentiment_charts/
├── README.md
└── requirements.txt
```

## 🧪 How to Run
```bash
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Launch Jupyter Notebook
jupyter notebook Sentiment_Analysis.ipynb

3️⃣ Run all cells sequentially
a) Load dataset

b) Preprocess text

c) Evaluate both models

b) Visualize results

Try creative analysis with your own text!
```

## 🧠 Key Learnings
- Hugging Face pipeline simplifies model inference for text classification.

- Transformers can generalize well to small datasets with minimal preprocessing.

- Model confidence and sentiment balance reveal how language models interpret tone.

- Fine-tuning enables domain-specific sentiment improvement.

## 🚀 Future Improvements
- Add SHAP or LIME for interpretability

- Deploy Streamlit sentiment dashboard

- Expand to multi-class emotion detection

- Fine-tune RoBERTa for lyrics/news data

## 🧑‍💻 Authors
- Viktor Kliufinskyi
- Priti Sagar
- Daniel Cortes

