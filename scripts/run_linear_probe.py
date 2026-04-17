#!/usr/bin/env python
import yaml
import sys
from data.counterfact_loader import CounterFactLoader
from models.linear_probe import LinearProbeDetector
from evaluation.metrics import print_metrics, save_results

def main():
    print("Loading config...")
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Load Data
    loader = CounterFactLoader(
        dataset_name_or_path=config['data']['dataset_name'], 
        split=config['data']['train_split']
    )
    dataset = loader.generate_labeled_pairs(limit=config['data'].get('max_samples', None))
    
    # Very basic train/test split (80/20) for demonstration
    split_idx = int(len(dataset) * 0.8)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    test_labels = [item['label'] for item in test_data]
    
    # 2. Initialize Model
    # Loop over all target layers to see which middle layer represents it best
    model_name = config['model_configs']['llama_base']
    batch_size = config['method1']['batch_size']
    
    overall_results = {}

    print(f"Initializing detector for {model_name}...")
    detector = LinearProbeDetector(model_name=model_name, target_layer=config['method1']['target_layers'][0], batch_size=batch_size)

    for layer in config['method1']['target_layers']:
        print(f"\\n--- Running Linear Probe for Layer {layer} ---")
        detector.target_layer = layer
        from sklearn.linear_model import LogisticRegression
        detector.classifier = LogisticRegression(max_iter=1000) # Reset classifier
        
        # 3. Train
        detector.train(train_data)
        
        # 4. Evaluate
        metrics = detector.evaluate(test_data, test_labels)
        print_metrics(metrics, method_name=f"Linear Probe (Layer {layer})")
        
        overall_results[f"layer_{layer}"] = metrics
        
    # 5. Save Results
    save_results(overall_results, "results/method1_metrics.json")
    print("Done!")

if __name__ == "__main__":
    main()
