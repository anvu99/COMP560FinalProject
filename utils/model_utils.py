import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

def load_hf_causal_model(model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Loads a Causal LM and Tokenizer optimized for hidden state extraction.
    Ensures memory efficiency using bfloat16.
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name} on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        output_hidden_states=True  # Crucial for Method 1
    )
    model.eval()
    return getattr(model, "module", model), tokenizer

def load_hf_sequence_classifier(model_name, num_labels=2, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Loads a Sequence Classification model (e.g. RoBERTa/DeBERTa) for Method 2.
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading classifier: {model_name} on {device}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    model.to(device)
    return model, tokenizer

def batch_process_data(data_list, batch_size):
    """
    Generator to yield batches of data from a list to prevent OOM errors.
    
    Args:
        data_list: List of data dictionaries.
        batch_size: Integer batch size.
        
    Yields:
        Lists of length up to `batch_size`.
    """
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]
