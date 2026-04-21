#!/usr/bin/env python
import argparse
import os
import yaml
import joblib
from data.counterfact_loader import CounterFactLoader
from models.linear_probe import LinearProbeDetector
from evaluation.metrics import print_metrics, save_results

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the Linear Probe conflict detector.")
    parser.add_argument("--output_dir", default="results/detectors/probe",
                        help="Directory to save the trained detector checkpoint and metrics.")
    args = parser.parse_args()

    print("Loading config...")
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Load Data
    loader = CounterFactLoader(
        dataset_name_or_path=config['data']['dataset_name'],
        split=config['data']['train_split']
    )
    dataset = loader.generate_labeled_pairs(limit=config['data'].get('max_samples', None))

    # Very basic train/test split (80/20)
    split_idx = int(len(dataset) * 0.8)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    test_labels = [item['label'] for item in test_data]

    # 2. Initialize Model
    model_name = config['model_configs']['llama_base']
    batch_size = config['method1']['batch_size']

    overall_results = {}
    best_f1 = -1.0
    best_layer = None
    best_classifier = None

    print(f"Initializing detector for {model_name}...")
    detector = LinearProbeDetector(
        model_name=model_name,
        target_layer=config['method1']['target_layers'][0],
        batch_size=batch_size
    )

    for layer in config['method1']['target_layers']:
        print(f"\n--- Running Linear Probe for Layer {layer} ---")
        detector.target_layer = layer
        from sklearn.linear_model import LogisticRegression
        detector.classifier = LogisticRegression(max_iter=config['method1'].get('max_iter', 1000))

        # 3. Train
        detector.train(train_data)

        # 4. Evaluate
        metrics = detector.evaluate(test_data, test_labels)
        print_metrics(metrics, method_name=f"Linear Probe (Layer {layer})")
        overall_results[f"layer_{layer}"] = metrics

        # Track best layer by F1
        if metrics.get("f1", 0.0) > best_f1:
            best_f1 = metrics["f1"]
            best_layer = layer
            import copy
            best_classifier = copy.deepcopy(detector.classifier)

    # 5. Save evaluation results
    os.makedirs("results", exist_ok=True)
    save_results(overall_results, "results/method1_metrics.json")

    # 6. Save best detector checkpoint for GUvICS integration
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "detector_probe.pkl")
    checkpoint = {
        "classifier": best_classifier,
        "target_layer": best_layer,
        "llm_model_name": model_name,
        "best_f1": best_f1,
    }
    joblib.dump(checkpoint, checkpoint_path)
    print(f"\nBest layer: {best_layer} (F1={best_f1:.4f})")
    print(f"Detector checkpoint saved to: {checkpoint_path}")
    print("Done!")

if __name__ == "__main__":
    main()
