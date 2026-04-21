#!/usr/bin/env python
import argparse
import os
import yaml
from data.counterfact_loader import CounterFactLoader
from models.encoder_nli import EncoderNLIDetector
from evaluation.metrics import print_metrics, save_results

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the Encoder NLI conflict detector.")
    parser.add_argument("--output_dir", default="results/detectors/nli/encoder",
                        help="Directory to save the fine-tuned encoder model and tokenizer.")
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

    split_idx = int(len(dataset) * 0.8)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    test_labels = [item['label'] for item in test_data]

    # 2. Initialize Model
    llm_name = config['model_configs']['llama_base']
    encoder_name = config['model_configs']['encoder_roberta']

    detector = EncoderNLIDetector(
        llm_model_name=llm_name,
        encoder_model_name=encoder_name,
        config_params=config['method2']
    )

    # 3. Train
    detector.train(train_data)

    # 4. Evaluate
    metrics = detector.evaluate(test_data, test_labels)
    print_metrics(metrics, method_name="Encoder NLI Fine-Tuning")

    # 5. Save evaluation results
    os.makedirs("results", exist_ok=True)
    save_results({"encoder_nli": metrics}, "results/method2_metrics.json")

    # 6. Save fine-tuned encoder checkpoint for GUvICS integration
    os.makedirs(args.output_dir, exist_ok=True)
    detector.encoder_model.save_pretrained(args.output_dir)
    detector.encoder_tokenizer.save_pretrained(args.output_dir)
    print(f"\nFine-tuned encoder saved to: {args.output_dir}")
    print("Done!")

if __name__ == "__main__":
    main()
