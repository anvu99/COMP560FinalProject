# Knowledge Conflict Detection in LLMs

## Overview  
This project studies how large language models internally represent **knowledge conflicts**. We evaluate three complementary methods to detect whether newly provided information contradicts a model’s parametric knowledge.

---

## Repository Structure  
- `configs/` – experiment configurations and hyperparameters  
- `data/` – dataset processing and storage  
- `models/` – implementations of the three methodologies  
- `evaluation/` – metrics and evaluation pipelines  
- `results/` – saved outputs and experiment results  
- `scripts/` – entry points for running experiments  
- `utils/` – helper functions and shared utilities  

---

## Dataset & Data Engineering  
We use the **CounterFact** dataset for both training and evaluation.

- **Positive samples (Conflict):** prompts paired with `target_new` (counterfactual edits)  
- **Negative samples (Non-Conflict):** prompts paired with `target_true` (factual answers aligned with model knowledge)  

This setup creates a binary classification task: **Conflict vs. Non-Conflict**.

---

## Methodologies  

### Method 1: Linear Probing (Primary)  
Extract hidden states $h_l$ from the LLM residual stream (final token). Focus on middle layers (e.g., 8–16). Train a logistic regression classifier on these representations.

### Method 2: Encoder Fine-Tuning  
Generate answers from the LLM, then train an encoder (e.g., RoBERTa) on:  
`[Generated Answer] [SEP] [New Knowledge]` → conflict probability.

### Method 3: Logit-Based Prompting (Baseline)  
Prompt the LLM directly and compute conflict probability from **Yes/No logits**.
