# Prompt Injection Classification with DistilBERT

Prompt Injection Classification aims to identify and categorize text samples that might include malicious or unintended prompts which could manipulate or mislead LLM.

## Structure

- `src/`: Contains all source code.
  - `models/`: Model scripts
  - `datasets/`: Dataset loaders and processors.
  - `utils/`: Utility scripts for metrics and additional functions.
- `config/`: Configuration files for models and datasets.


### Install packages

```bash
git clone https://github.com/karthickai/llm_detect.git
cd llm_detect
pip install -r requiremnts.txt
```

### Run the model
Modify inference script for your need!
```bash
python inference.py
```

### Training the Model

Make sure to edit the configurations in `config/` directory to suit your dataset and model parameters.

```bash
python -m llmdetect.models.sequence_classification.train
```
