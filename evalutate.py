from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader

# Load dataset
dataset = load_dataset('csv', data_files='./dataset/data.csv')

# Load model and tokenizer
model_path = "model.safetensors"
tokenizer = AutoTokenizer.from_pretrained('tokenizer.json')
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Preprocess function
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess_function)
tokenized_dataset = tokenized_dataset.with_format("torch")
keep_columns = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
eval_columns = tokenized_dataset.column_names
# Columns to remove
e_columns = [col for col in eval_columns if col not in keep_columns]
tokenized_dataset = tokenized_dataset.map(remove_columns=e_columns)

# Inference
dataloader = DataLoader(tokenized_dataset, batch_size=32, shuffle=False)
model.eval()
all_predictions = []
with torch.no_grad():
    for batch in dataloader:
        inputs = {key: batch[key] for key in ['input_ids', 'attention_mask']}
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        all_predictions.extend(predictions.numpy())

# Map predictions
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
mapped_predictions = [label_map[pred] for pred in all_predictions]

print(mapped_predictions)
