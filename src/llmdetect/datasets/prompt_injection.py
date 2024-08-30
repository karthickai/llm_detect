from datasets import load_dataset
from transformers import DistilBertTokenizer

# Todo
# Add method combine multiple prompt dataset
# create Tranfer learning dataset load 15% of previous dataset in to training

class PromptInjectionDataset:
    def __init__(self, dataset_name, tokenizer_name='distilbert-base-uncased', data_dir="../../../data"):
        self.dataset_name = dataset_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.data_dir = data_dir

    def tokenize_data(self, examples):
        return self.tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    def process_labels(self, examples):
        examples['label'] = [int(label) for label in examples['label']]
        return examples

    def load(self):
        dataset = load_dataset(self.dataset_name, cache_dir=self.data_dir)
        dataset = dataset.map(self.tokenize_data, batched=True)
        dataset = dataset.map(self.process_labels, batched=True)  # Add label processing
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        return dataset
