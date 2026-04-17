import json
import random
from datasets import load_dataset

class CounterFactLoader:
    def __init__(self, dataset_name_or_path="NeelNanda/counterfact-tracing", split="train"):
        """
        Loads the CounterFact dataset for conflict generation.
        
        Args:
            dataset_name_or_path: HuggingFace hub path or local file.
            split: Which split to load (usually there is just 'train' in NeelNanda's version, 
                   so we might need to do our own train/test split later).
        """
        self.dataset_name = dataset_name_or_path
        self.split = split
        print(f"Loading CounterFact dataset from {dataset_name_or_path}...")
        self.raw_data = load_dataset(dataset_name_or_path, split=split)
        
    def generate_labeled_pairs(self, limit=None):
        """
        Processes raw CounterFact entries into labeled Conflict/Non-Conflict pairs.
        
        For each entry:
        - Positive Sample (Conflict, Label=1): prompt + target_new
        - Negative Sample (Non-Conflict, Label=0): prompt + target_true
        
        Returns:
            list of dicts containing the processed samples.
        """
        processed_data = []
        count = 0
        
        for item in self.raw_data:
            if limit is not None and count >= limit:
                break
                
            rewrite = item['requested_rewrite']
            prompt = rewrite['prompt']
            target_true = rewrite['target_true']['str']
            target_new = rewrite['target_new']['str']
            subject = rewrite['subject']
            
            # Format the prompt by injecting the subject
            if "{}" in prompt:
                formatted_prompt = prompt.replace("{}", subject)
            else:
                formatted_prompt = prompt
                
            # Create Positive Sample (Conflict, Label 1)
            processed_data.append({
                "prompt": formatted_prompt,
                "fact": target_new,
                "label": 1,
                "type": "conflict"
            })
            
            # Create Negative Sample (Non-Conflict, Label 0)
            processed_data.append({
                "prompt": formatted_prompt,
                "fact": target_true,
                "label": 0,
                "type": "non-conflict"
            })
            
            count += 1
            
        return processed_data
        
if __name__ == "__main__":
    # Simple test
    loader = CounterFactLoader()
    samples = loader.generate_labeled_pairs(limit=2)
    for s in samples:
        print(s)
