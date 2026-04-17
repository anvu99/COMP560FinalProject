import torch
from vllm import LLM, SamplingParams
from transformers import Trainer, TrainingArguments
from datasets import Dataset as HFDataset
from models.base_detector import BaseConflictDetector
from utils.model_utils import load_hf_sequence_classifier

class EncoderNLIDetector(BaseConflictDetector):
    def __init__(self, llm_model_name, encoder_model_name, config_params=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.llm_model_name = llm_model_name
        self.encoder_model_name = encoder_model_name
        self.device = device
        
        self.config = config_params or {}
        
        # Load the classifier
        self.encoder_model, self.encoder_tokenizer = load_hf_sequence_classifier(
            self.encoder_model_name, 
            num_labels=2, 
            device=self.device
        )
        self.is_trained = False

    def generate_answers(self, dataset):
        """
        Uses vLLM to quickly generate the LLM's parametric answers to the prompts.
        This provides the baseline belief to compare against the new facts.
        """
        print(f"Initializing vLLM with {self.llm_model_name} for generation...")
        # Reduce gpu_memory_utilization and limit context length because Llama 3.1 has a 131k context 
        # which requires 16GB of KV cache by default. We only need short contexts for CounterFact.
        llm = LLM(
            model=self.llm_model_name, 
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.5, 
            max_model_len=2048
        )
        sampling_params = SamplingParams(temperature=0.0, max_tokens=30)
        
        prompts = [item['prompt'] for item in dataset]
        print(f"Generating answers for {len(prompts)} prompts...")
        outputs = llm.generate(prompts, sampling_params)
        
        # Map generated text back into the dataset
        processed_dataset = []
        for i, out in enumerate(outputs):
            gen_text = out.outputs[0].text.strip()
            item = dataset[i].copy()
            item['generated_answer'] = gen_text
            processed_dataset.append(item)
            
        print("Generation complete.")
        return processed_dataset

    def extract_features(self, dataset):
        """
        Checks if generations exist. If not, generates them.
        Formats data into HuggingFace dataset for Sequence Classification Trainer.
        """
        if len(dataset) > 0 and 'generated_answer' not in dataset[0]:
             dataset = self.generate_answers(dataset)
             
        # Format as proper NLI premise-hypothesis pairs:
        # "[Prompt + Generated Answer] [SEP] [Prompt + New Fact]"
        # Including the prompt in both sides gives the encoder relational context,
        # so it knows both answers belong to the same question and can detect contradiction.
        # e.g. "The mother tongue of X is French [SEP] The mother tongue of X is English"
        texts = [
            f"{item['prompt']} {item['generated_answer']} "
            f"{self.encoder_tokenizer.sep_token} "
            f"{item['prompt']} {item['fact']}"
            for item in dataset
        ]
        labels = [item['label'] for item in dataset]
        
        def tokenize_function(examples):
            return self.encoder_tokenizer(
                examples['text'], 
                padding="max_length", 
                truncation=True, 
                max_length=self.config.get('max_length', 128)
            )

        hf_dataset = HFDataset.from_dict({"text": texts, "label": labels})
        hf_dataset = hf_dataset.map(tokenize_function, batched=True)
        # Remove string columns to avoid Trainer issues
        hf_dataset = hf_dataset.remove_columns(["text"])
        hf_dataset.set_format("torch")
        
        return hf_dataset

    def train(self, train_data, output_dir="results/encoder_nli"):
        train_dataset = self.extract_features(train_data)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=float(self.config.get('learning_rate', 2e-5)),
            per_device_train_batch_size=self.config.get('batch_size', 16),
            num_train_epochs=self.config.get('epochs', 3),
            weight_decay=0.01,
            eval_strategy="no",        # renamed from evaluation_strategy in transformers >= 4.46
            save_strategy="no"         # Turned off to prevent I/O quota errors on home directory
        )

        trainer = Trainer(
            model=self.encoder_model,
            args=training_args,
            train_dataset=train_dataset
        )

        print(f"Starting finetuning on {self.encoder_model_name}...")
        trainer.train()
        self.is_trained = True

    def predict(self, test_data):
        probas = self.predict_proba(test_data)
        import numpy as np
        return np.argmax(probas, axis=1)

    def predict_proba(self, test_data):
        if not self.is_trained:
            print("Warning: Classifier is not trained. Running zero-shot.")
            
        test_dataset = self.extract_features(test_data)
        
        trainer = Trainer(model=self.encoder_model)
        predictions = trainer.predict(test_dataset)
        
        # Apply softmax to raw logits
        logits = torch.tensor(predictions.predictions)
        probas = torch.nn.functional.softmax(logits, dim=-1).numpy()
        
        return probas
