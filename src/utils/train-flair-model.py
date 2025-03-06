import pandas as pd
import re
from sklearn.model_selection import train_test_split
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

# Load dataset
df = pd.read_csv('D:/python_proj/wcloudgui/dataset/test.tsv', sep='\t')

# Clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

df['cleaned_text'] = df['Text'].apply(clean_text)

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Save datasets as TSV
train_df[['cleaned_text', 'Sentiment']].to_csv('train_data.tsv', sep='\t', index=False, header=False)
val_df[['cleaned_text', 'Sentiment']].to_csv('val_data.tsv', sep='\t', index=False, header=False)
test_df[['cleaned_text', 'Sentiment']].to_csv('test_data.tsv', sep='\t', index=False, header=False)

# Load dataset in Flair
columns = {0: 'text', 1: 'label'}
corpus: Corpus = ColumnCorpus('./', columns,
                              train_file='train_data.tsv',
                              dev_file='val_data.tsv',
                              test_file='test_data.tsv')

# Create label dictionary
label_dict = corpus.make_label_dictionary()

# Choose embeddings
word_embeddings = WordEmbeddings('id')
document_embeddings = DocumentRNNEmbeddings([word_embeddings], hidden_size=256, rnn_layers=1, rnn_type='LSTM')

# Create text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

# Train the model
trainer = ModelTrainer(classifier, corpus)
trainer.train('resources/taggers/sentiment_model',
              learning_rate=0.1,
              mini_batch_size=32,
              anneal_factor=0.5,
              patience=5,
              max_epochs=150)

# Evaluate the model
trainer.test(corpus.test)

# Save the model
classifier.save('resources/taggers/sentiment_model/final-model.pt')

# Load and use the model
classifier = TextClassifier.load('resources/taggers/sentiment_model/final-model.pt')

# Example prediction
sentence = Sentence("Saya sangat senang dengan produk ini!")
classifier.predict(sentence)

# Print predicted label
print(sentence.labels)
