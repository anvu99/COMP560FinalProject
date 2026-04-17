import numpy as np
import torch
from tqdm import tqdm
from models.base_detector import BaseConflictDetector
from utils.model_utils import load_hf_causal_model, batch_process_data
from evaluation.metrics import compute_metrics

class LogitBaselineDetector(BaseConflictDetector):
    def __init__(self, model_name, prompt_template, default_threshold=0.5, batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.threshold = default_threshold
        self.batch_size = batch_size
        self.device = device
        
        self.model, self.tokenizer = load_hf_causal_model(self.model_name, self.device)
        self.is_trained = False
        
        # Identify the exact token IDs for "Yes" and "No"
        # We check both capitalized and lowercase to be safe, but usually it's "Yes"
        self.yes_tokens = self._get_tokens(["Yes", " Yes", "yes", " yes"])
        self.no_tokens = self._get_tokens(["No", " No", "no", " no"])
        
    def _get_tokens(self, words):
        """Helper to safely map strings to standard vocab tokens"""
        tokens = []
        for w in words:
            # Add prefix space for Llama tokenizer quirks if necessary, though auto handles it
            t_ids = self.tokenizer(w, add_special_tokens=False).input_ids
            if len(t_ids) == 1:
                tokens.append(t_ids[0])
        return list(set(tokens))

    def extract_features(self, dataset):
        """
        Runs the prompt format and calculates soft logits.
        Does not return feature vectors, returns direct probability arrays
        because Method 3 is zero-shot.
        """
        all_probs = []
        all_labels = []

        print(f"Extracting soft logits for {len(dataset)} samples...")
        
        with torch.no_grad():
            for batch in tqdm(list(batch_process_data(dataset, self.batch_size))):
                inputs_text = [
                    self.prompt_template.format(context=item['prompt'], new_fact=item['fact']) 
                    for item in batch
                ]
                labels = [item['label'] for item in batch]
                
                inputs = self.tokenizer(
                    inputs_text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(self.device)
                
                outputs = self.model(**inputs)
                
                # Get the logits for the very last token
                # Shape: (batch_size, vocab_size)
                sequence_lengths = inputs['attention_mask'].sum(dim=1) - 1
                batch_size = outputs.logits.shape[0]
                final_logits = outputs.logits[torch.arange(batch_size, device=self.device), sequence_lengths, :]
                
                # Aggregate probabilities for Yes vs No tokens
                for b_idx in range(batch_size):
                    yes_logits = final_logits[b_idx, self.yes_tokens].max() # Take best "yes" token
                    no_logits = final_logits[b_idx, self.no_tokens].max()   # Take best "no" token
                    
                    # Convert to a mini softmax over just these two concepts
                    stacked = torch.stack([no_logits, yes_logits])
                    probs = torch.nn.functional.softmax(stacked, dim=0)
                    
                    # probability of 'Yes' (Conflict)
                    prob_conflict = probs[1].item()
                    all_probs.append([1.0 - prob_conflict, prob_conflict])
                
                all_labels.extend(labels)
                
        return np.array(all_probs), np.array(all_labels)

    def train(self, train_data):
        """
        Method 3 requires no parameter updates. 
        'Training' finds the optimal decision threshold.
        """
        print("Tuning threshold on train set...")
        probas, y_true = self.extract_features(train_data)
        
        best_f1 = 0
        best_threshold = 0.5
        
        # Grid search threshold
        for t in np.arange(0.1, 1.0, 0.05):
            preds = (probas[:, 1] >= t).astype(int)
            metrics = compute_metrics(y_true, preds)
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_threshold = t
                
        print(f"Optimal threshold found: {best_threshold:.2f} (F1: {best_f1:.4f})")
        self.threshold = best_threshold
        self.is_trained = True

    def predict(self, test_data) -> np.ndarray:
        probas, _ = self.extract_features(test_data)
        return (probas[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, test_data) -> np.ndarray:
        probas, _ = self.extract_features(test_data)
        return probas
