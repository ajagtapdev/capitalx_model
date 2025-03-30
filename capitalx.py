# fine-tuning script for CapitalX credit card recommendation system
# uses meta-llama/Meta-Llama-3-8B model with:
# - modal for cloud training
# - peft for parameter-efficient fine-tuning
# - 8-bit quantization to fit model in GPU memory
# - learning rate of 5e-6
# - 2 epochs of training
#
# modal run capitalx.py

import json
import os
import modal
from preprocessing import (
    create_prompt, create_target, load_data, preprocess_data, 
    collate_fn, generate_recommendation, CardDataset
)

# Global variable for model name
MODEL_NAME = "google/gemma-2-2b"

# Setting up correct path for HF token to ensure authentication works consistently
HF_TOKEN_DIR = "/root/.cache/huggingface"
HF_TOKEN_PATH = f"{HF_TOKEN_DIR}/token"

# --- Modal Setup for Distributed Training with Multiple Containers ---

app = modal.App("credit_card_recommender")

# Create a Modal image with all required packages and CUDA setup
image = (modal.Image.debian_slim()
    .run_commands([
        # Add NVIDIA CUDA repository
        "apt-get update && apt-get install -y wget gnupg",
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb",
        "dpkg -i cuda-keyring_1.0-1_all.deb",
        "apt-get update",
        # Install CUDA toolkit
        "apt-get install -y cuda-toolkit-12-2",
        # Set CUDA environment variables
        "echo 'export CUDA_HOME=/usr/local/cuda' >> /etc/profile",
        "echo 'export PATH=$PATH:$CUDA_HOME/bin' >> /etc/profile",
        "echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64' >> /etc/profile",
        # Create triton directory
        "mkdir -p /root/.triton/autotune",
        "chmod -R 777 /root/.triton"
    ])
    .pip_install([
        "torch>=2.0.0", 
        "transformers>=4.30.0", 
        "accelerate>=0.20.0", 
        "bitsandbytes>=0.39.0", 
        "peft>=0.4.0", 
        "deepspeed>=0.9.0",
        "huggingface_hub", 
        "python-dotenv"
    ])
    # Add all local files LAST
    .add_local_file(
        "capitalx_training_data.json", 
        remote_path="/root/capitalx_training_data.json"
    )
    .add_local_file(
        "preprocessing.py",
        remote_path="/root/preprocessing.py"
    )
    .add_local_file(
        "ds_config.json", 
        remote_path="/root/ds_config.json"
    )
    .add_local_python_source("preprocessing")
)

model_volume = modal.Volume.from_name("models", create_if_missing=True)

@app.function(
    image=image,
    timeout=300,
    secrets=[modal.Secret.from_name("HUGGING_FACE_TOKEN")],
    volumes = {"/model": model_volume}
)


def test_create_prompt():
    """Test the prompt creation function."""
    sample = {
        "cards": [
            {
                "name": "Card Test",
                "reward_plan": {
                    "base_rate": 1.0,
                    "categories": {"groceries": 3.0}
                },
                "apr": 15.0,
                "credit_limit": 5000
            }
        ],
        "transaction": {
            "product": "Test Product",
            "category": "groceries",
            "vendor": "Test Vendor",
            "price": 50.0
        },
        "best_card": "Card Test",
        "explanation": "Highest grocery rewards."
    }
    prompt = create_prompt(sample)
    expected = (
        "Cards:\n"
        "Card Test: APR 15.0%, Credit Limit 5000, Rewards: base 1.0, groceries 3.0%\n"
        "Transaction:\n"
        "Product: Test Product, Category: groceries, Vendor: Test Vendor, Price: 50.0\n"
        "Output: "
    )
    assert prompt == expected, f"Prompt does not match expected. Got: {prompt}"
    print("test_create_prompt passed.")

def test_create_target():
    """Test the target creation function."""
    sample = {
        "best_card": "Card Test",
        "explanation": "Highest grocery rewards."
    }
    target = create_target(sample)
    expected = "Best card: Card Test. Explanation: Highest grocery rewards."
    assert target == expected, f"Target does not match expected. Got: {target}"
    print("test_create_target passed.")

def test_inference(model_path="fine_tuned_model/best"):
    """Test the trained model with a sample transaction."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig
    
    hf_token = os.environ.get("HUGGING_FACE_TOKEN")
    
    try:
        # For LoRA fine-tuned models
        print(f"Loading LoRA adapter from {model_path}")
        
        # First load base model (for adapter)
        config = PeftConfig.from_pretrained(model_path)
        print(f"Detected base model: {config.base_model_name_or_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path, 
            token=hf_token
        )
        
        # Load base model with 8-bit quantization
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
            token=hf_token
        )
        
        # Load adapter on top of base model
        model = PeftModel.from_pretrained(base_model, model_path)
        
    except Exception as e:
        print(f"Failed to load as PEFT model, trying standard model: {e}")
        # Try loading as regular model
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto",
            token=hf_token
        )
    
    model.eval()
    
    # Sample test case
    sample = {
        "cards": [
            {
                "name": "Freedom Flex",
                "reward_plan": {
                    "base_rate": 1.0,
                    "categories": {"dining": 3.0, "groceries": 5.0}
                },
                "apr": 15.0,
                "credit_limit": 5000
            },
            {
                "name": "Sapphire Preferred",
                "reward_plan": {
                    "base_rate": 1.0,
                    "categories": {"travel": 4.0, "dining": 3.0}
                },
                "apr": 18.0,
                "credit_limit": 10000
            }
        ],
        "transaction": {
            "product": "Dinner",
            "category": "dining",
            "vendor": "Restaurant",
            "price": 75.0
        }
    }
    
    result = generate_recommendation(model, tokenizer, sample)
    print("Sample Transaction:")
    print(create_prompt(sample))
    print("\nModel Recommendation:")
    print(result)

# Create separate containers for different training stages
@app.function(
    image=image, 
    timeout=36000,
    gpu="H100:8",
    memory=128000,
    secrets=[modal.Secret.from_name("HUGGING_FACE_TOKEN")],
    volumes = {"/model": model_volume}
)
def process_and_train(model_name=MODEL_NAME):
    """Preprocess data and run model training using all 8 GPUs."""
    import os
    import torch
    import json
    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    # Set CUDA environment variables explicitly
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["PATH"] = f"{os.environ['PATH']}:/usr/local/cuda/bin"
    os.environ["LD_LIBRARY_PATH"] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:/usr/local/cuda/lib64"
    
    # Authentication first
    hf_token = os.environ.get("HUGGING_FACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGING_FACE_TOKEN environment variable is not set")
    
    # Login and set up cache
    print("Logging in to Hugging Face Hub...")
    login(token=hf_token)
    
    # Create token file
    os.makedirs(HF_TOKEN_DIR, exist_ok=True)
    with open(HF_TOKEN_PATH, 'w') as f:
        f.write(hf_token)
    print(f"HF token file created at {HF_TOKEN_PATH}")
    
    # Set environment variables for auth and caching
    os.environ["HUGGING_FACE_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    os.environ["HF_TOKEN"] = hf_token
    
    # Set up cache directories on the persistent volume
    model_cache_dir = "/model/transformers_cache"
    os.makedirs(model_cache_dir, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = model_cache_dir
    os.environ["HF_HOME"] = "/model/huggingface_home"
    os.environ["HF_DATASETS_CACHE"] = "/model/datasets_cache"
    
    # Check if model is already cached
    cached_model_path = f"{model_cache_dir}/{model_name.replace('/', '--')}"
    if os.path.exists(cached_model_path):
        print(f"Found cached model at {cached_model_path}")
    else:
        print(f"Model not found in cache, will download to {model_cache_dir}")
    
    # Output more detailed GPU information
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    total_gpu_memory = 0
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
        print(f"GPU {i}: {gpu_name} with {gpu_memory:.2f} GB memory")
        total_gpu_memory += gpu_memory
    
    print(f"Total GPU memory across all devices: {total_gpu_memory:.2f} GB")
    
    # Process data with more explicit instruction
    print("Loading and preprocessing data")
    train_data, val_data = load_data("capitalx_training_data.json")
    print(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples")
    
    print(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        cache_dir=model_cache_dir
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add instruction prefix to each sample to ensure output format consistency
    def add_instruction_to_prompt(sample):
        original_prompt = create_prompt(sample)
        instruction = "You are a credit card recommendation assistant. Analyze the cards and transaction, then recommend the BEST card with a clear explanation.\n\n"
        return instruction + original_prompt
    
    # Modify train and val data with instruction prefix
    for sample in train_data:
        sample["enhanced_prompt"] = add_instruction_to_prompt(sample)
    
    for sample in val_data:
        sample["enhanced_prompt"] = add_instruction_to_prompt(sample)
    
    # Modify the preprocessing function to use the enhanced prompt if available
    def preprocess_with_instruction(data, tokenizer, max_length=256):
        processed = []
        for sample in data:
            if "enhanced_prompt" in sample:
                prompt = sample["enhanced_prompt"]
            else:
                prompt = create_prompt(sample)
            target = create_target(sample)
            full_text = prompt + target
            tokenized = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors="pt")
            tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
            processed.append(tokenized)
        return processed
    
    # Use the modified preprocessing function
    train_processed = preprocess_with_instruction(train_data, tokenizer)
    val_processed = preprocess_with_instruction(val_data, tokenizer)
    
    print("Data preprocessing complete with instruction prefix")
    
    # Output directories
    output_dir = "/model/model_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure QLoRA (Quantized LoRA) for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    # Option 1: Using Hugging Face Trainer with DeepSpeed for multi-GPU training
    print(f"Loading model {model_name} with 4-bit quantization")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",  # Let accelerate handle device mapping
        token=hf_token,
        cache_dir=model_cache_dir
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Target key modules for reasoning tasks
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    ))
    model.print_trainable_parameters()
    
    # Optimize batch size for 8 GPUs - increase effective batch size
    # With 8 GPUs we can process more data in parallel
    # Create DataCollator
    class CardDataCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            
        def __call__(self, batch):
            return collate_fn(batch, self.tokenizer)
    
    # Set up training arguments for Trainer with DeepSpeed
    # Adjust batch sizes for 8 GPUs
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=2,  # Keep per-device batch size modest
        per_device_eval_batch_size=4,   # Can be larger for evaluation
        gradient_accumulation_steps=4,  # Reduced from 8 since we have more GPUs
        learning_rate=5e-6,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        report_to="none",
        deepspeed="/root/ds_config.json",  # Use DeepSpeed config
        ddp_find_unused_parameters=False,
        local_rank=-1,  # Let DeepSpeed handle the local rank
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_processed,
        eval_dataset=val_processed,
        data_collator=CardDataCollator(tokenizer),
    )
    
    # Train the model
    print("Starting multi-GPU training")
    trainer.train()
    
    # Save best model
    trainer.save_model(f"{output_dir}/best")
    tokenizer.save_pretrained(f"{output_dir}/best")
    print("Training complete!")
    
    return output_dir


@app.function(
    image=image,
    gpu="H100:8",
    timeout=600,
    secrets=[modal.Secret.from_name("HUGGING_FACE_TOKEN")],
    volumes = {"/model": model_volume}
)

@app.function(
    image=image,
    gpu="H100:1",  # One GPU is sufficient for evaluation
    memory=64000,
    timeout=3600,
    secrets=[modal.Secret.from_name("HUGGING_FACE_TOKEN")],
    volumes = {"/model": model_volume}
)
def run_evaluation(model_path="/model/model_output/best", model_name=MODEL_NAME):
    """Run model evaluation on a smaller GPU instance."""
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig
    from huggingface_hub import login
    
    # Authentication setup
    hf_token = os.environ.get("HUGGING_FACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGING_FACE_TOKEN environment variable is not set")
    
    login(token=hf_token)
    
    # Set up cache directories
    model_cache_dir = "/model/transformers_cache"
    os.makedirs(model_cache_dir, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = model_cache_dir
    
    # Load the fine-tuned model
    try:
        # Try loading as a PEFT/LoRA model first
        config = PeftConfig.from_pretrained(model_path)
        print(f"Loading adapter on top of {config.base_model_name_or_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path,
            token=hf_token
        )
        
        # Load base model with 8-bit quantization for efficiency
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            load_in_8bit=True,
            device_map="auto",
            token=hf_token
        )
        
        # Load adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        
    except Exception as e:
        print(f"Failed to load as PEFT model: {e}, trying standard model")
        # Fall back to loading as a standard model
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            load_in_8bit=True,
            device_map="auto",
            token=hf_token
        )
    
    model.eval()
    
    # Test cases
    test_cases = [
        {
            "cards": [
                {
                    "name": "Freedom Flex",
                    "reward_plan": {
                        "base_rate": 1.0,
                        "categories": {"dining": 3.0, "groceries": 5.0}
                    },
                    "apr": 15.0,
                    "credit_limit": 5000
                },
                {
                    "name": "Sapphire Preferred",
                    "reward_plan": {
                        "base_rate": 1.0,
                        "categories": {"travel": 4.0, "dining": 3.0}
                    },
                    "apr": 18.0,
                    "credit_limit": 10000
                }
            ],
            "transaction": {
                "product": "Dinner",
                "category": "dining",
                "vendor": "Restaurant",
                "price": 75.0
            }
        },
        {
            "cards": [
                {
                    "name": "Freedom Flex",
                    "reward_plan": {
                        "base_rate": 1.0,
                        "categories": {"dining": 3.0, "groceries": 5.0}
                    },
                    "apr": 15.0,
                    "credit_limit": 5000
                },
                {
                    "name": "Sapphire Preferred",
                    "reward_plan": {
                        "base_rate": 1.0,
                        "categories": {"travel": 4.0, "dining": 3.0}
                    },
                    "apr": 18.0,
                    "credit_limit": 10000
                }
            ],
            "transaction": {
                "product": "Groceries",
                "category": "groceries",
                "vendor": "Supermarket",
                "price": 120.0
            }
        }
    ]
    
    results = {}
    for i, test_case in enumerate(test_cases):
        print(f"\nEvaluating test case {i+1}")
        result = generate_recommendation(model, tokenizer, test_case)
        print(f"Result: {result}")
        results[f"test_case_{i+1}"] = result
    
    return results

@app.local_entrypoint()
def main(model_name=MODEL_NAME):  # Remove test_auth and test_gpus parameters
    """Main entry point for the script."""
    # Load token from environment (only for local testing)
    from dotenv import load_dotenv
    load_dotenv()
    
    print(f"Starting multi-GPU fine-tuning workflow for {model_name}")
    
    # Training with multi-GPU setup
    print("Step 1: Preprocessing and training with multiple GPUs...")
    model_path = process_and_train.remote(model_name)
    print(f"Training complete. Model saved at: {model_path}")
    
    # Evaluation
    print("Step 2: Evaluating model...")
    result = run_evaluation.remote(f"{model_path}/best", model_name)
    print("Evaluation complete!")
    print("Sample recommendation:")
    print(result)

if __name__ == "__main__":
    main()
