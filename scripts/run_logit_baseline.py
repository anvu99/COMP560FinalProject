#!/usr/bin/env python
import yaml
from data.counterfact_loader import CounterFactLoader
from models.logit_baseline import LogitBaselineDetector
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
    
    split_idx = int(len(dataset) * 0.8)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    test_labels = [item['label'] for item in test_data]
    
    # 2. Initialize Model
    model_name = config['model_configs']['llama_base']
    
    detector = LogitBaselineDetector(
        model_name=model_name, 
        prompt_template=config['method3']['prompt_template'],
        batch_size=config['method1']['batch_size']
    )
    
    # 3. Train (Tunes Threshold over Train Data)
    detector.train(train_data)
    
    # 4. Evaluate
    metrics = detector.evaluate(test_data, test_labels)
    print_metrics(metrics, method_name=f"Logit Prompting (Threshold {detector.threshold:.2f})")
    
    # 5. Save Results
    save_results({
        "logit_baseline": metrics, 
        "optimal_threshold": detector.threshold
    }, "results/method3_metrics.json")
    print("Done!")

if __name__ == "__main__":
    main()
