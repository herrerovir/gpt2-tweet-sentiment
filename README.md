# 🐤💭 Tweet Sentiment Classification with GPT-2

This project fine-tunes a pre-trained transformer model, **GPT-2**, to classify tweet sentiments into three categories: **Negative**, **Neutral**, and **Positive**. The model was trained and evaluated on a labeled dataset of tweets, with the entire workflow executed in **Google Colab** using a **Tesla T4 GPU** to accelerate training and inference. The goal is to create a lightweight, accurate sentiment classifier that can be used to analyze social media content in real time.

## 🗃️ Repository Structure

```plaintext
gpt2-tweet-sentiment/
│
├── data/                                      # Tweet dataset
│   ├── raw/                                   # Raw tweet data
│   └── processed/                             # Cleaned data
│
├── figures/                                   # Visualizations
│   └── gpt2-model-confusion-matrix.png        # Model confusion matrix
│
├── models/                                    # Trained GPT-2 models
│   └── gpt2-final-model/                      # Final saved model and tokenizer
│
├── notebooks/                                 # Notebooks
│   └── gpt2-finetune-tweet-sentiment.ipynb    # End-to-end pipeline
│
├── results/                                   # Model output
│   ├── metrics/                               # Evaluation results
│   │   └── gpt2-model-evaluation-metrics.txt
│   └── predictions/                           # Inference results
│       └── predictions_output.txt                       
│
├── config.py                                  # Google Drive & Colab folder setup
├── requirements.txt                           # Dependencies
└── README.md                                  # Project documentation
```

## 📘 Project Overview

- **Introduction** – Fine-tuned **GPT-2** to classify tweets into **Negative**, **Neutral**, and **Positive** sentiment categories.

- **Data Cleaning** – Removed null values, duplicates, mentions, URLs, and extra whitespace for cleaner inputs.

- **Tokenization and Data Collation** – Tokenized tweets using the GPT-2 tokenizer, with padding dynamically handled during batching.

- **Model Setup and Fine-Tuning** – Loaded `GPT2ForSequenceClassification` with 3 output labels. Trained over 5 epochs using Hugging Face’s `Trainer`.

- **Training Configuration** – Optimized training with batch size 8, learning rate 2e-5, and automatic model checkpointing and evaluation.

- **Evaluation Metrics** – Used accuracy and weighted F1-score, with confusion matrix and classification report to analyze performance.

- **Inference Pipeline** – Created a `TextClassificationPipeline` to predict sentiment from real tweets, along with confidence scores.

- **Conclusion** – Delivered a robust sentiment analysis model ready for use in real-time applications like social media monitoring or customer feedback analysis.

## ⚙️ Dependencies

This project requires the following libraries:

```bash
pip install -r requirements.txt
```

* **Python**
* **PyTorch**
* **Transformers (Hugging Face)**
* **Scikit-learn**
* **Numpy**
* **Matplotlib**
* **Pandas**

## ▶️ How to Run the Project

### Option 1: Run Locally with GPU

1. Clone this repository:

   ```bash
   git clone https://github.com/herrerovir/gpt2-tweet-sentiment
   ```

2. Navigate to the project directory:

   ```bash
   cd gpt2-tweet-sentiment
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the notebook or script to train and test the model:

   ```bash
   jupyter notebook
   ```

### Option 2: Run on Google Colab (Recommended)

1. Open a new Google Colab notebook.

2. Clone the repository inside the notebook:

   ```python
   !git clone https://github.com/herrerovir/gpt2-tweet-sentiment
   ```

3. Navigate to the cloned folder and open the notebook `gpt2-finetune-tweet-sentiment.ipynb`.

4. Switch runtime to GPU (preferably **Tesla T4**) for faster training.

5. Follow the notebook to fine-tune GPT-2 and perform inference.

## 📂 Model Files

The fine-tuned GPT-2 model is saved directly to your **Google Drive** under `models/gpt2/gpt2-final-model`. It includes:

* Model weights
* Tokenizer files (`vocab.json`, `merges.txt`, config files)

These allow you to reload the model for inference or further fine-tuning without retraining from scratch.

## 📊 Model Performance

After training for 5 epochs, the GPT-2 model achieved:

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 79.37% |
| F1 Score  | 79.34% |
| Eval Loss | 0.6867 |

### 🔍 Class Performance Breakdown

| Class    | Precision | Recall | F1-Score |
| -------- | --------- | ------ | -------- |
| Negative | 0.81      | 0.78   | 0.79     |
| Neutral  | 0.76      | 0.76   | 0.76     |
| Positive | 0.82      | 0.85   | 0.84     |

These results show the model generalizes well and maintains balance across all sentiment classes.

## 🔮 Inference Examples

The model accurately classifies tweet sentiments with confidence scores:

- **Input:** *"The food was hot and delicious."* **Prediction:** Positive (Confidence: 99.93%)

- **Input:** *"Ugh, my flight got delayed again."* **Prediction:** Negative (Confidence: 99.95%)

- **Input:** *"Heading to the grocery store, then back to work."* **Prediction:** Neutral (Confidence: 99.57%)

- **Input:** *"Lost all my work because of a crash. Fantastic."* **Prediction:** Positive (Confidence: 52.06%) ⚠️ (sarcasm not detected)

These highlight both the strengths and limitations of the model, especially when sarcasm is involved.

## 📋 Results

The GPT-2 model proves effective for sentiment classification on social media text. With nearly **80% accuracy and F1 score**, and consistent per-class performance, it's a strong baseline for real-world applications. It performs especially well on clearly positive or negative tweets, but can be improved to better detect sarcasm or subtle tones.

## 🙌 Acknowledgments

Built with [Hugging Face Transformers](https://huggingface.co/transformers/), [PyTorch](https://pytorch.org/), and [Scikit-learn](https://scikit-learn.org/). Trained using free GPU resources via Google Colab.
