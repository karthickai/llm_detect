from datasets import load_dataset, concatenate_datasets
from transformers import DistilBertTokenizer
import os
# Todo
# Add method combine multiple prompt dataset
# create Tranfer learning dataset load 15% of previous dataset in to training


class PromptInjectionDataset:
    def __init__(self, dataset_cfg, tokenizer_name='distilbert-base-uncased', data_dir="data"):
        self.dataset_names = dataset_cfg['name']
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.data_dir = data_dir
        self.data_cfg = dataset_cfg
        print(dataset_cfg)

    def tokenize_data(self, examples):
        return self.tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    def filter_labels(self, example):
        # Return True only if the label is 0 or 1
        return int(example['label']) <= 1

    def process_labels(self, examples):
        examples['label'] = [int(label) for label in examples['label']]
        return examples

    def load(self):
        all_datasets = []
        for name in self.dataset_names:
            if "local#" in name:
                dataset = load_dataset("arrow", data_files={'train': self.data_cfg[name]['train'], 'test': self.data_cfg[name]['test']}, cache_dir="data")
            elif "JailbreakV-28K" in name:
                dataset = load_dataset(name, 'JailBreakV_28K', cache_dir="data")
                dataset = dataset['JailBreakV_28K'].map(lambda x: {'text': x['jailbreak_query'], 'label': 1})
                dataset = dataset.train_test_split(test_size=0.1)
            else:
                dataset = load_dataset(name, cache_dir="data")
            dataset = dataset.filter(self.filter_labels)
            dataset = dataset.map(self.tokenize_data, batched=True)
            dataset = dataset.map(self.process_labels, batched=True)
            dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            all_datasets.append(dataset)
            print(f"Loaded {name} train dataset with {len(dataset['train'])} examples and {len(dataset['test'])} test examples")
        expected_splits = all_datasets[0].keys()
        print(f"Expected splits: {expected_splits}")
        assert all(all_datasets[0].keys() == d.keys() for d in all_datasets), "All datasets should have the same splits"
        combined_dataset = {split: concatenate_datasets([d[split] for d in all_datasets]) for split in all_datasets[0].keys()}
        return combined_dataset

