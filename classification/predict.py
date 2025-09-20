# classification/predict.py
import torch
import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from utils.helpers import load_data, save_data # Adjusted import path

print("DEBUG: predict.py - Module loaded.")

def predict_distress(config): # Accepts full config
    print("DEBUG: predict.py - Entering predict_distress function.")
    data_config = config['data'] # Use 'data' section now
    class_config = config['classification']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEBUG: predict.py - Using device: {device}")

    print(f"DEBUG: predict.py - Loading tokenizer from {class_config['tokenizer_path']}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(class_config['tokenizer_path'])
    
    print(f"DEBUG: predict.py - Initializing model from {class_config['model_name']} and loading weights from {class_config['model_path']}...")
    model = DistilBertForSequenceClassification.from_pretrained(class_config['model_name'], num_labels=2)
    model.load_state_dict(torch.load(class_config['model_path'], map_location=device))
    model.to(device)
    model.eval()
    print("DEBUG: predict.py - Model loaded and set to evaluation mode.")

    print(f"DEBUG: predict.py - Loading processed data from {data_config['processed_data_path']} for prediction...")
    df = load_data(data_config['processed_data_path'])
    texts = df['cleaned_text'].tolist()
    print(f"DEBUG: predict.py - Loaded {len(texts)} texts for prediction.")
    
    probabilities = []
    with torch.no_grad():
        for i in range(0, len(texts), class_config['batch_size']):
            batch_texts = texts[i:i + class_config['batch_size']]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=class_config['max_length']).to(device)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            probabilities.extend(probs)
            
    df['distress_probability'] = probabilities
    save_data(df, class_config['predictions_output_path'])
    print(f"DEBUG: predict.py - Predictions saved to {class_config['predictions_output_path']}.")
    print("DEBUG: predict.py - Exiting predict_distress function.")
