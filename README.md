<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis on Movie Reviews</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            padding: 20px;
            background-color: #f9f9f9;
        }
        h1, h2, h3 {
            color: #333;
        }
        code {
            background: #eee;
            padding: 2px 4px;
            border-radius: 4px;
        }
        pre {
            background: #333;
            color: #fff;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #fff;
        }
        table, th, td {
            border: 1px solid #ddd;
            text-align: center;
            padding: 10px;
        }
        th {
            background: #555;
            color: white;
        }
    </style>
</head>
<body>

    <h1>Sentiment Analysis on Movie Reviews</h1>

    <h2>Overview</h2>
    <p>This project builds a <strong>sentiment analysis model</strong> using <strong>LSTM (Long Short-Term Memory)</strong> networks to classify movie reviews as <strong>positive or negative</strong>.</p>

    <h2>Dependencies</h2>
    <p>Install the required Python libraries:</p>
    <pre><code>pip install tensorflow pandas numpy scikit-learn nltk</code></pre>

    <h2>Dataset</h2>
    <p>The dataset consists of:</p>
    <ul>
        <li><strong>text</strong>: The movie review.</li>
        <li><strong>label</strong>: Sentiment label (1 for positive, 0 for negative).</li>
    </ul>

    <h2>Data Cleaning</h2>
    <p>To preprocess the data, we apply several text-cleaning techniques:</p>

    <h3>1. Remove Special Characters</h3>
    <pre><code>import re
def remove_splchar(content):
    return re.sub('\[[^&@#!]]*/]',' ',content)
    </code></pre>

    <h3>2. Remove URLs</h3>
    <pre><code>def remove_url(content):
    return re.sub(r"http\S+|www\S+|https\S+", ' ', content)
    </code></pre>

    <h3>3. Remove Stopwords</h3>
    <pre><code>from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

def remove_stopword(content):
    data = [word.strip().lower() for word in content.split() if word.strip().lower() not in stop_words and word.strip().lower().isalpha()]
    return " ".join(data)
    </code></pre>

    <h3>4. Expand Contractions</h3>
    <pre><code>def contract_extract(content):
    content = re.sub(r"won\'t", "would not", content)
    content = re.sub(r"can\'t", "can not", content)
    content = re.sub(r"don\'t", "do not", content)
    content = re.sub(r"shouldn\'t", "should not", content)
    content = re.sub(r"needn\'t", "need not", content)
    content = re.sub(r"hasn\'t", "has not", content)
    content = re.sub(r"haven\'t", "have not", content)
    content = re.sub(r"weren\'t", "were not", content)
    content = re.sub(r"mightn\'t", "might not", content)
    content = re.sub(r"didn\'t", "did not", content)
    content = re.sub(r"aren\'t", "are not", content)
    content = re.sub(r"isn\'t", "is not", content)
    content = re.sub(r"wouldn\'t", "would not", content)
    content = re.sub(r"willn\'t", "will not", content)
    content = re.sub(r"\'t", "not", content)
    content = re.sub(r"\'ve", "have", content)
    content = re.sub(r"\'m", "am", content)
    return content
    </code></pre>

    <h3>5. Stemming</h3>
    <pre><code>from nltk.stem import SnowballStemmer
def stemming(data):
    snowball = SnowballStemmer("english")
    return ' '.join([snowball.stem(word) for word in data.split()])
    </code></pre>

    <h3>6. Full Data Cleaning Function</h3>
    <pre><code>def data_cleaning(content):
    content = remove_splchar(content)
    content = remove_url(content)
    content = contract_extract(content)
    content = remove_stopword(content)
    content = stemming(content)
    return content
    </code></pre>

    <h2>Model Architecture</h2>
    <p>The model consists of:</p>
    <ul>
        <li><strong>Embedding Layer</strong>: Converts words into dense vectors.</li>
        <li><strong>SpatialDropout1D</strong>: Reduces overfitting.</li>
        <li><strong>LSTM Layer</strong>: Captures sequential dependencies.</li>
        <li><strong>Dense Layer</strong>: Outputs a probability using sigmoid activation.</li>
    </ul>

    <h2>Results</h2>
    <table>
        <tr>
            <th>Label</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>Support</th>
        </tr>
        <tr>
            <td>0 (Negative)</td>
            <td>0.84</td>
            <td>0.87</td>
            <td>0.86</td>
            <td>2495</td>
        </tr>
        <tr>
            <td>1 (Positive)</td>
            <td>0.87</td>
            <td>0.84</td>
            <td>0.85</td>
            <td>2505</td>
        </tr>
        <tr>
            <td colspan="5"><strong>Overall Metrics</strong></td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td colspan="4">0.86</td>
        </tr>
        <tr>
            <td>Macro Avg</td>
            <td>0.86</td>
            <td>0.86</td>
            <td>0.86</td>
            <td>5000</td>
        </tr>
        <tr>
            <td>Weighted Avg</td>
            <td>0.86</td>
            <td>0.86</td>
            <td>0.86</td>
            <td>5000</td>
        </tr>
    </table>

    <h2>Future Improvements</h2>
    <ul>
        <li>Use <strong>pre-trained embeddings</strong> like GloVe or Word2Vec.</li>
        <li>Fine-tune hyperparameters and increase epochs.</li>
        <li>Experiment with <strong>Bidirectional LSTMs</strong> or <strong>CNN+LSTM</strong> models.</li>
    </ul>

</body>
</html>
