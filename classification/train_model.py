# classification/train_model.py
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils.helpers import load_data, ensure_dir # Adjusted import path
from accelerate import Accelerator

print("DEBUG: train_model.py - Module loaded.")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def train_classification_model(config): # Accepts full config
    print("DEBUG: train_model.py - Entering train_classification_model function.")
    data_config = config['data'] # Use 'data' section now
    class_config = config['classification']

    print(f"DEBUG: train_model.py - Loading processed data from {data_config['processed_data_path']}...")
    df = load_data(data_config['processed_data_path'])
    print(f"DEBUG: train_model.py - Data loaded. Shape: {df.shape}")

    train_df, test_df = train_test_split(df, test_size=data_config['test_size'], random_state=data_config['random_state'], stratify=df['label'])
    print(f"DEBUG: train_model.py - Data split into train ({len(train_df)}) and test ({len(test_df)}).")
    
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    print(f"DEBUG: train_model.py - Initializing tokenizer from {class_config['model_name']}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(class_config['model_name'])
    print(f"DEBUG: train_model.py - Initializing model from {class_config['model_name']}...")
    model = DistilBertForSequenceClassification.from_pretrained(class_config['model_name'], num_labels=2)

    def tokenize(batch):
        return tokenizer(batch['cleaned_text'], padding=True, truncation=True, max_length=class_config['max_length'])

    print("DEBUG: train_model.py - Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    print("DEBUG: train_model.py - Datasets tokenized and formatted.")

    training_args = TrainingArguments(
        output_dir=class_config['output_dir'],
        num_train_epochs=class_config['num_epochs'],
        per_device_train_batch_size=class_config['batch_size'],
        per_device_eval_batch_size=class_config['batch_size'],
        learning_rate=class_config['learning_rate'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )
    print("DEBUG: train_model.py - TrainingArguments initialized.")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    print("--- Starting model training ---")
    print(f"DEBUG: train_model.py - Training epochs: {class_config['num_epochs']}")
    trainer.train()
    print("DEBUG: train_model.py - Model training completed.")
    
    ensure_dir(class_config['tokenizer_path'])
    tokenizer.save_pretrained(class_config['tokenizer_path'])
    print(f"DEBUG: train_model.py - Tokenizer saved to {class_config['tokenizer_path']}.")
    torch.save(model.state_dict(), class_config['model_path'])
    print(f"DEBUG: train_model.py - Model weights saved to {class_config['model_path']}.")
    print("DEBUG: train_model.py - Exiting train_classification_model function.")
