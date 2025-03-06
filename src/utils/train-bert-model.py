from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
import os

# Define the columns in the CSV files
columns = {0: "text", 1: "label"}

# Path to the dataset folder
data_folder = os.path.abspath(r'd:\python_proj\wcloudgui\dataset')

# Paths to the train, dev, and test files
train_file = os.path.join(data_folder, 'train.csv')
dev_file = os.path.join(data_folder, 'dev.csv')
test_file = os.path.join(data_folder, 'test.csv')

# Check if the files exist
for file_path, file_name in [(train_file, "train"), (dev_file, "dev"), (test_file, "test")]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_name} file not found: {file_path}")

# Debugging: Print sample lines (limited to 5)
def print_sample(file_path, name):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"\nSample from {name}.csv ({len(lines)} lines):")
        print(''.join(lines[:5]))

print_sample(train_file, "train")
print_sample(dev_file, "dev")
print_sample(test_file, "test")

# Create a corpus using the CSV files
corpus: Corpus = CSVClassificationCorpus(
    data_folder,
    columns,
    train_file='train.csv',
    dev_file='dev.csv',
    test_file='test.csv',
    label_type='label',
    delimiter=',',  
    skip_header=True,
    encoding='utf-8'
)

# Debugging: Check the number of sentences in each split
print(f"Train set: {len(corpus.train)} sentences")
print(f"Dev set: {len(corpus.dev)} sentences")
print(f"Test set: {len(corpus.test)} sentences")

# Check for empty sentences
for split, name in [(corpus.train, "train"), (corpus.dev, "dev"), (corpus.test, "test")]:
    empty_count = sum(1 for s in split if len(s) == 0)
    if empty_count > 0:
        print(f"Warning: {empty_count} empty sentences found in {name} set!")

# Create the label dictionary
label_dict = corpus.make_label_dictionary(label_type='label')

# Initialize the document embeddings using a pre-trained transformer model
document_embeddings = TransformerDocumentEmbeddings('indobenchmark/indobert-base-p1')

# Create the text classifier
classifier = TextClassifier(
    document_embeddings,
    label_dictionary=label_dict,
    label_type='label',
    multi_label=False
)

# Initialize the model trainer
trainer = ModelTrainer(classifier, corpus)

# Train the model
trainer.train(
    base_path='resources/taggers/sentiment-indonesian',
    learning_rate=3e-5,
    mini_batch_size=16,
    max_epochs=10
)
