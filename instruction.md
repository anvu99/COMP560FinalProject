Project Task Instruction: Knowledge Conflict Detection Pipeline
Project Objective: To develop and evaluate methods capable of detecting when new contextually provided knowledge conflicts with a Large Language Model's internal parametric knowledge. This detection module serves as the foundational conflict signal for State-of-the-art knowledge editing architecture, ensuring that the model only suppresses old weights when genuinely contradictory information is introduced.
1. Dataset & Data Engineering
Primary Dataset: CounterFact (utilized for both training and testing phases).
Positive Samples (Conflict): Use the standard counterfactual prompts and the target_new edits provided in the dataset.
Negative Samples (Non-Conflict): We will implement negative sampling by pairing the prompts with their corresponding target_true values. Since the model's parametric knowledge aligns with the true facts, this serves as our baseline for non-conflicting knowledge integration.

2. Methodology & Implementation Scope
We will evaluate three distinct approaches to measure and detect internal knowledge conflicts.
Method 1: Linear Probing on the Residual Stream (Primary Focus)
Concept: LLMs internally register the signal of knowledge conflict within their residual streams, typically peaking in the middle layers.
Action Items:
Pass the input prompts paired with the new knowledge through the target LLM.
Extract the hidden states ($h_l$) from the residual stream at the final token position. Focus primarily on the middle layers (e.g., layers 8 through 16 for an 8B parameter model), as MLP and Self-Attention activations in later layers yield diminishing returns for this specific signal.
Train a logistic regression classifier on these activation vectors to perform binary classification (Conflict vs. Non-Conflict).
Method 2: Encoder Fine-Tuning via Answer Generation
Concept: Framing conflict detection as a Natural Language Inference (NLI) task by comparing the model's parametric output against the new provided knowledge.
Action Items:
Run the LLM over the CounterFact dataset in a zero-shot setting to extract its generated answers based purely on its pre-trained weights.
Fine-tune an encoder-only model (e.g., RoBERTa, DeBERTa) on a sequence classification task.
Input format: [Generated Answer] + [SEP] + [New Knowledge]
Output: Probability of contradiction/conflict.
Method 3: Logit-Based Prompting (Baseline)
Concept: Evaluating the LLM's explicit "self-awareness" of knowledge conflicts through direct prompting.
Action Items:
Construct a standardized prompt template: [Context/New Fact] + Question: Does this knowledge conflict with your internal knowledge? Answer Yes or No.
Run the prompt through the model and extract the raw logits for the first generated output token (specifically the logits for the Yes and No tokens).
Apply a softmax function over these two logits to calculate a normalized conflict probability score.
Establish a decision threshold to classify the final output.

