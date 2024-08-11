import sys
import os
import hydra
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from llmdetect.datasets.prompt_injection import PromptInjectionDataset
from llmdetect.utils.metrics import compute_metrics

sys.path.append(os.path.abspath('../../../'))

@hydra.main(config_path='../../../../config', config_name='config.yml')
def train_model(cfg: DictConfig):
    dataset = PromptInjectionDataset(cfg.dataset.name).load()

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name,
        num_labels=cfg.model.num_labels
    )

    train_dataset = dataset['train'].remove_columns(["text"]).with_format("torch")
    test_dataset = dataset['test'].remove_columns(["text"]).with_format("torch")

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

if __name__ == "__main__":
    train_model()
