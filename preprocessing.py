import json
import torch
from torch.utils.data import Dataset

def create_prompt(sample):
    """
    Format:
      Cards:
      <card details line by line>
      Transaction:
      Product: <product>, Category: <category>, Vendor: <vendor>, Price: <price>
      Output:
    """
    prompt = "Cards:\n"
    for card in sample["cards"]:
        card_details = (
            f"{card['name']}: APR {card['apr']}%, Credit Limit {card['credit_limit']}, "
            f"Rewards: base {card['reward_plan']['base_rate']}, " +
            ", ".join([f"{cat} {rate}%" for cat, rate in card["reward_plan"]["categories"].items()])
        )
        prompt += card_details + "\n"
    
    t = sample["transaction"]
    prompt += "Transaction:\n"
    prompt += f"Product: {t['product']}, Category: {t['category']}, Vendor: {t['vendor']}, Price: {t['price']}\n"
    prompt += "Output: "
    
    return prompt

def create_target(sample):
    """
    Creates a target string from a training sample.
    Format:
      Best card: <best_card>. Explanation: <explanation>
    """
    return f"Best card: {sample['best_card']}. Explanation: {sample['explanation']}"

def load_data(json_path="capitalx_training_data.json", train_split=0.8):
    """Loads training data from a JSON file and splits into train/val sets."""
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Split into train and validation sets
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data

def preprocess_data(data, tokenizer, max_length=256):
    """
    For each sample, concatenates the prompt and target to form the full text,
    then tokenizes the result.
    """
    processed = []
    for sample in data:
        prompt = create_prompt(sample)
        target = create_target(sample)
        full_text = prompt + target
        tokenized = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors="pt")
        # Remove extra batch dimension
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
        processed.append(tokenized)
    
    # Save tokenizer for later use if needed
    tokenizer.save_pretrained("/model/tokenizer")
    
    return processed

def collate_fn(batch, tokenizer):
    """Pads batch samples to the maximum length in the batch."""
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": input_ids.clone()
    }

class CardDataset(Dataset):
    """Custom dataset for credit card recommendation data."""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def generate_recommendation(model, tokenizer, sample, max_new_tokens=50, temperature=0.3, do_sample=False):
    """Generate a card recommendation for a transaction using the trained model."""
    prompt = create_prompt(sample)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return result.strip() 