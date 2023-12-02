# Importing necessary libraries
!pip install langdetect
import pandas as pd
import json
from langdetect import detect
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Loading the review dataset
with open('/content/drive/MyDrive/Pers IDS/IIT Docs/Yelp Dataset/yelp_academic_dataset_review.json') as f:
    df_review = pd.DataFrame([json.loads(line) for line in f.readlines()])

# Checking the initial length of the dataset
print("Initial length of the review dataset:", len(df_review))

# Removing unnecessary columns from df_review
df_review.drop(['review_id', 'user_id', 'useful', 'funny', 'cool', 'date'], axis=1, inplace=True)

# Loading the business dataset
with open('/content/drive/MyDrive/Pers IDS/IIT Docs/Yelp Dataset/yelp_academic_dataset_business.json') as f:
    df_business = pd.DataFrame([json.loads(line) for line in f.readlines()])

# Filtering out rows with null 'categories' and focusing on restaurants
df_business_new = df_business[df_business['categories'].notnull()]
df_business_new = df_business_new[df_business_new['categories'].str.contains('Restaurants')]

# Keeping only necessary columns
df_business_new = df_business_new[['business_id', 'categories']]

# Merging the two dataframes
df_joined = pd.merge(df_review, df_business_new, on='business_id', how='inner')

# Renaming and dropping columns
df_joined.rename(columns={'text': 'restaurant_reviews'}, inplace=True)
df_joined.drop('business_id', axis=1, inplace=True)

# Removing rows with 3-star ratings
df_joined = df_joined[df_joined['stars'] != 3]

# Sampling, rows from the DataFrame
df_joined = df_joined.sample(n=10000, random_state=1)

# Detecting language and filtering out non-English reviews
df_joined = df_joined[df_joined['restaurant_reviews'].apply(lambda x: detect(x) == 'en')]

# Removing NaN values and duplicate rows
df_joined.dropna(inplace=True)
df_joined.drop_duplicates(inplace=True)

# Removing 3-star ratings
df_joined = df_joined[df_joined['stars'] != 3]

# Sentiment labeling
df_joined['sentiment'] = df_joined['stars'].apply(lambda x: 1 if x > 3 else 0)
df_joined.drop('stars', axis=1, inplace=True)

# Checking the updated length of the dataset
print("Updated length of the dataset:", len(df_joined))

# Plotting sentiment distribution
sns.countplot(x='sentiment', data=df_joined)
plt.xlabel('Sentiment Label')
plt.ylabel('Number of Reviews')
plt.title('Sentiment Distribution in Yelp Reviews')
plt.show()

# Balancing the dataset by resampling
negative_reviews = df_joined[df_joined['sentiment'] == 0]
positive_reviews = df_joined[df_joined['sentiment'] == 1]

negative_reviews = negative_reviews.sample(len(positive_reviews), replace=True)
df_balanced = pd.concat([positive_reviews, negative_reviews])

# Loading spaCy English model
nlp = spacy.load('en_core_web_sm')

# Function to preprocess text
def text_preprocessing(text):
    doc = nlp(text)
    return ' '.join([token.text.lower() for token in doc if token.is_alpha and token.text.lower() not in STOP_WORDS])

# Applying the preprocessing to the reviews
df_balanced['cleaned_reviews'] = df_balanced['restaurant_reviews'].apply(text_preprocessing)

# Resetting the index
df_clean = df_balanced.reset_index(drop=True)

# Convert DataFrame to CSV

df_clean.to_csv('cleaned_reviews.csv', index=False)
###########-----------------###############

# Load Data for SVM from CSV
df_clean = pd.read_csv('cleaned_reviews.csv')

# Prepare Data for SVM
X = df_clean['cleaned_reviews']
y = df_clean['sentiment']

# Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Split Data into Training and Test Sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train Linear SVM Model

from sklearn.svm import LinearSVC

svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

# Evaluate the Model
from sklearn.metrics import classification_report, accuracy_score

y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix for Linear SVM Model')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.xticks(ticks=[0.5, 1.5], labels=['Negative', 'Positive'])
plt.yticks(ticks=[0.5, 1.5], labels=['Negative', 'Positive'])
plt.show()
