# Yelp Review Sentiment Analysis Using BERT

## Project Overview
This project focuses on sentiment analysis using a pre-trained BERT model (`nlptown/bert-base-multilingual-uncased-sentiment`) to evaluate customer reviews. The goal is to automatically classify the sentiment of text reviews from a popular review site like Yelp. The sentiment classification is based on a five-point scale, where 1 indicates a very negative sentiment, and 5 indicates a very positive sentiment. 

### 1. Install and Import Dependencies
Before you start, make sure to install all the necessary libraries:

```python
!pip install torch transformers requests beautifulsoup4 pandas numpy
```

These libraries are essential for the following tasks:

- **PyTorch:** A deep learning framework used to handle tensor operations and manage the model.  
- **transformers:** A library from Hugging Face that provides pre-trained models and tokenizers.  
- **requests:** For making HTTP requests to get data from websites.  
- **beautifulsoup4:** To parse HTML and extract data.  
- **pandas and numpy:** For data manipulation and handling.

### 2. Instantiate the Model
Here, we load the tokenizer and model from Hugging Face:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
```
These lines of code initialize the BERT tokenizer and model that are pre-trained on sentiment analysis tasks.

### 3. Encode and Calculate Sentiment
To analyze the sentiment of a review, encode the text and predict the sentiment:

```python
tokens = tokenizer.encode('It was awesome, I loved it', return_tensors='pt')
result = model(tokens)

sentiment_score = int(torch.argmax(result.logits)) + 1
print(sentiment_score)
```

The sentiment score ranges from 1 (very negative) to 5 (very positive).
The tokenized text is processed by the model to produce logits, which are then used to predict the sentiment class.

### 4. Collect Reviews
You can collect customer reviews from a webpage using the requests and BeautifulSoup libraries:

```python
import requests
import re
from bs4 import BeautifulSoup

r = requests.get('https://www.yelp.com/biz/mcdonalds-new-york-386?osq=McDonald%27s')
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class': regex})
```

This code fetches the reviews from the provided Yelp page and identifies text elements containing customer reviews.

### 5. DataFrame Creation and CSV Export

The extracted reviews can be stored in a pandas DataFrame and then exported as a CSV file:

```python
import pandas as pd

reviews = [result.get_text() for result in results]
df = pd.DataFrame(reviews, columns=['Review'])
df['Sentiment'] = df['Review'].apply(lambda review: int(torch.argmax(model(tokenizer.encode(review, return_tensors='pt')).logits)) + 1)

df.to_csv('yelp_reviews.csv', index=False)
```

This creates a CSV file with the customer reviews and their corresponding sentiment scores, which can be used for further analysis.

## Conclusion

The project successfully demonstrates the use of a pre-trained BERT model for sentiment analysis on text data. By leveraging the power of transformers and PyTorch, the project provides a scalable solution for automating the sentiment classification of customer reviews. This approach can be applied to various applications, including customer feedback analysis, market research, and social media monitoring.  
Overall, this project highlights the effectiveness of using advanced NLP models like BERT for real-world sentiment analysis tasks, providing insights that can drive better decision-making and improve customer experiences.


For any questions or further assistance, please feel free to contact me.

Dhananjaya Mudunkotuwa   
dhananjayamudunkotuwa1998@gmail.com
