# Sentiment Analysis on Movie Reviews

## Overview
This project applies **LSTM-based sentiment analysis** to classify movie reviews as **positive** or **negative**.

## Dataset
- **Text**: The movie review.
- **Label**: 1 (positive), 0 (negative).

## Installation
Ensure you have Python installed, then run:

```sh
pip install tensorflow pandas numpy scikit-learn nltk
```

## Data Preprocessing
- **Removing special characters**
- **Removing stopwords**
- **Stemming text**

## Model Architecture
- **Embedding Layer**: Word vector representations
- **SpatialDropout1D**: Reduces overfitting
- **LSTM Layer**: Captures sequential patterns
- **Dense Layer**: Sigmoid activation for classification

## Training & Evaluation
Model trained using **binary cross-entropy loss** and **Adam optimizer**.

### Results:
| Label    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|----------|
| Negative | 0.84      | 0.87   | 0.86     | 2495     |
| Positive | 0.87      | 0.84   | 0.85     | 2505     |
| **Accuracy** | **0.86** | **0.86** | **0.86** | **5000** |

## Future Improvements
- Use **pre-trained embeddings** (GloVe, Word2Vec)
- Optimize hyperparameters and increase epochs
- Experiment with **Bidirectional LSTMs** or **CNN+LSTM**

## Usage
Run the following command to train the model:

```sh
python train.py
```

## License
This project is licensed under the MIT License.
