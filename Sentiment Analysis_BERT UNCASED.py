# Importing necessary libraries
!pip
install
langdetect
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


from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Token length distribution
token_lens = []
for txt in df_clean['cleaned_reviews']:
    tokens = tokenizer.encode(txt, max_length=512)
    token_lens.append(len(tokens))

# Visualize token length distribution
sns.histplot(token_lens, bins=30)
plt.xlim([0, 512])
plt.xlabel('Token count')
plt.show()

from sklearn.model_selection import train_test_split

# Splitting the data
train_texts, temp_texts, train_labels, temp_labels = train_test_split(df_clean['cleaned_reviews'], df_clean['sentiment'],
                                                                      test_size=0.3, random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels,
                                                                  test_size=0.5, random_state=42)

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

MAX_SEQ_LENGTH = 260  # Chosen sequence length

def convert_to_dataset(texts, labels):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,                       # Text to encode
                            add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
                            max_length = MAX_SEQ_LENGTH,
                            padding = 'max_length',     # Pad & truncate all sentences
                            truncation = True,
                            return_attention_mask = True,
                            return_tensors = 'pt'       # Return PyTorch tensors
                       )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels.tolist())  # Convert labels to a list before converting to tensor

    return TensorDataset(input_ids, attention_masks, labels)

# Convert datasets
train_dataset = convert_to_dataset(train_texts, train_labels)
val_dataset = convert_to_dataset(val_texts, val_labels)
test_dataset = convert_to_dataset(test_texts, test_labels)

from transformers import BertForSequenceClassification, AdamW

# Load BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 2, # Binary classification
    output_attentions = False,
    output_hidden_states = False,
)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=3e-5)

from transformers import get_linear_schedule_with_warmup

EPOCHS = 2

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = len(train_dataloader) * EPOCHS)

from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

# Create DataLoader for the test dataset
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)

# Move model to evaluation mode
model.eval()

# Tracking variables
predictions , true_labels = [], []

# Predict
for batch in test_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch

    # Telling the model not to compute or store gradients, saving memory and speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs.logits

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)

# Flatten the predictions and true labels
flat_predictions = np.concatenate(predictions, axis=0)
flat_true_labels = np.concatenate(true_labels, axis=0)

# Convert logits to predicted class (0 or 1)
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

# Calculate the accuracy
accuracy = (flat_predictions == flat_true_labels).mean()
print('Test Accuracy: %.3f' % accuracy)


import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

# Assuming model, test_dataloader are already defined and model is on the correct device ('cuda' or 'cpu')

# Put model in evaluation mode
model.eval()

# Initialize lists to store predictions and true labels
predictions, true_labels = [], []

# Evaluate the model
with torch.no_grad():
    for batch in test_dataloader:
        # Add batch to the device
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Get predictions
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs.logits

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

# Flatten the predictions and true labels
flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = np.concatenate(true_labels, axis=0)

# Calculate the accuracy
accuracy = accuracy_score(flat_true_labels, flat_predictions)
print(f'Accuracy: {accuracy}')

# Generate classification report
class_report = classification_report(flat_true_labels, flat_predictions)
print("Classification Report:\n", class_report)

# Generate confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Given confusion matrix
conf_matrix = [[2, 1169], [0, 1129]]

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.xticks(ticks=[0.5, 1.5], labels=['Negative', 'Positive'])
plt.yticks(ticks=[0.5, 1.5], labels=['Negative', 'Positive'])
plt.show()

