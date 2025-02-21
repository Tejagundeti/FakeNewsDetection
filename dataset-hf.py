from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the LIAR dataset from Hugging Face
dataset = load_dataset("liar", trust_remote_code=True)

# Keep only 'label' and 'statement' columns
dataset = dataset.remove_columns([col for col in dataset["train"].column_names if col not in ["label", "statement"]])

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["statement"], padding="max_length", truncation=True)

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert dataset to PyTorch tensors
tokenized_datasets = tokenized_datasets.remove_columns(["statement"])  # Remove text, keep tokenized version
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 🚀 Split train dataset into train/validation sets (90% Train, 10% Validation)
train_size = int(0.9 * len(tokenized_datasets["train"]))
train_dataset = tokenized_datasets["train"].select(range(train_size))
eval_dataset = tokenized_datasets["train"].select(range(train_size, len(tokenized_datasets["train"])))

# 🚀 Test dataset (if available)
test_dataset = tokenized_datasets["test"] if "test" in tokenized_datasets else None

# Load BERT model (6 labels for LIAR dataset)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro")  # Multi-class
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# 🚀 Train the model
trainer.train()

# 🚀 Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)
