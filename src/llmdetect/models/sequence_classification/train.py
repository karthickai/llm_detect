import sys
import os
import hydra
from omegaconf import DictConfig, ListConfig
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from llmdetect.datasets.prompt_injection import PromptInjectionDataset
from llmdetect.utils.metrics import compute_metrics

@hydra.main(config_path='../../../../config', config_name='config.yml')
def train_model(cfg: DictConfig):
    print("cfg.dataset.name", cfg.dataset.name)
    if isinstance(cfg.dataset.name, ListConfig):
        dataset = PromptInjectionDataset(cfg.dataset).load()
    else:
        dataset = PromptInjectionDataset([cfg.dataset]).load() 
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name,
        num_labels=cfg.model.num_labels
    )

    train_dataset = dataset['train'].remove_columns(["text"]).with_format("torch")
    test_dataset = dataset['test'].remove_columns(["text"]).with_format("torch")
    print(f"Unique labels {test_dataset['label'].unique()}")
    print(f"Total length of the train dataset: {len(train_dataset)}")
    print(f"Total length of the test dataset: {len(test_dataset)}")
    exit()

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=cfg.model.epochs,
        per_device_train_batch_size=cfg.model.batch_size,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    trainer.save_model("combined_dataset_model")

if __name__ == "__main__":
    train_model()
