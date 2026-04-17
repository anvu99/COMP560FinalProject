import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from models.base_detector import BaseConflictDetector
from utils.model_utils import load_hf_causal_model, batch_process_data

class LinearProbeDetector(BaseConflictDetector):
    def __init__(self, model_name, target_layer, batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.target_layer = target_layer
        self.batch_size = batch_size
        self.device = device
        
        # Will hold the sklearn classifier
        self.classifier = LogisticRegression(max_iter=1000)
        
        # Load the model & tokenizer immediately
        self.model, self.tokenizer = load_hf_causal_model(self.model_name, self.device)
        self.is_trained = False

    def extract_features(self, dataset):
        """
        Extract hidden states from the target layer at the final token position.
        
        Dataset format expected: list of dicts with 'prompt' and 'fact'.
        Format passed to model: "{prompt} {fact}"
        """
        all_activations = []
        all_labels = []

        print(f"Extracting features from Layer {self.target_layer} for {len(dataset)} samples...")
        
        # Turn off gradients for inference
        with torch.no_grad():
            for batch in tqdm(list(batch_process_data(dataset, self.batch_size))):
                inputs_text = [f"{item['prompt']} {item['fact']}" for item in batch]
                labels = [item['label'] for item in batch]
                
                # Tokenize
                inputs = self.tokenizer(
                    inputs_text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(self.device)
                
                # Pass through model
                outputs = self.model(**inputs)
                
                # outputs.hidden_states is a tuple of (layers + 1)
                # Shape of each element: (batch_size, sequence_length, hidden_dimension)
                hidden_states = outputs.hidden_states[self.target_layer]
                
                # Get the activation of the last token for each sequence in the batch
                sequence_lengths = inputs['attention_mask'].sum(dim=1) - 1 # 0-indexed pos of last token
                
                batch_size = hidden_states.shape[0]
                last_token_activations = hidden_states[torch.arange(batch_size, device=self.device), sequence_lengths]
                
                # Move to CPU for scikit-learn
                all_activations.append(last_token_activations.cpu().float().numpy())
                all_labels.extend(labels)

        # Concatenate all batches
        X = np.vstack(all_activations)
        y = np.array(all_labels)
        
        return X, y

    def train(self, train_data):
        """Trains Logistic Regression classifier on the extracted features."""
        X_train, y_train = self.extract_features(train_data)
        
        print("Training Logistic Regression classifier...")
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        print(f"Training complete. Classes: {self.classifier.classes_}")

    def predict(self, test_data) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Classifier not trained yet!")
        X_test, _ = self.extract_features(test_data)
        return self.classifier.predict(X_test)

    def predict_proba(self, test_data) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Classifier not trained yet!")
        X_test, _ = self.extract_features(test_data)
        return self.classifier.predict_proba(X_test)
