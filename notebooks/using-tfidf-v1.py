import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df = pd.read_csv('C:\\Users\\ASUS\\OneDrive\\Documents\\Fake_News_Classifier\\data\\WELFake_Dataset.csv')

def clean_text(text):
    text = str(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df['combined'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
df['clean_text'] = df['combined'].apply(clean_text)

X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, 'fake_news_classifier.joblib')   


