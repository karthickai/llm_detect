from datasets import load_dataset
from transformers import DistilBertTokenizer

class PromptInjectionDataset:
    def __init__(self, dataset_name, tokenizer_name='distilbert-base-uncased'):
        self.dataset_name = dataset_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)

    def tokenize_data(self, examples):
        return self.tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    def load(self):
        dataset = load_dataset(self.dataset_name)
        dataset = dataset.map(self.tokenize_data, batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        return dataset
