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

# Create a Modal image with all required packages
image = modal.Image.debian_slim().pip_install([
    "torch>=2.0.0", 
    "transformers>=4.30.0", 
    "accelerate>=0.20.0", 
    "bitsandbytes>=0.39.0", 
    "peft>=0.4.0", 
    "deepspeed>=0.9.0",  # Upgrade to a newer version
    "huggingface_hub", 
    "python-dotenv"
])

# Add data file to the image
image = image.add_local_file(
    "capitalx_training_data.json", 
    remote_path="/root/capitalx_training_data.json"
)

# Add preprocessing module to the image
image = image.add_local_file(
    "preprocessing.py",
    remote_path="/root/preprocessing.py"
)

# Add this line after creating the initial image
image = image.add_local_python_source("preprocessing")

# After the initial image creation, add this directory creation step
image = image.run_commands([
    "mkdir -p /root/.triton/autotune",
    "chmod -R 777 /root/.triton"
])

# Add DeepSpeed configuration file optimized for 8 GPUs
deepspeed_config = {
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param": {
            "device": "cpu"
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "gather_16bit_weights_on_model_save": True
    },
    "train_batch_size": "auto",  # Let DeepSpeed calculate the global batch size
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 4,  # Adjusted for 8 GPUs
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-6,
            "weight_decay": 0.01,
            "eps": 1e-8
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 5e-6,
            "warmup_num_steps": 100
        }
    },
    "steps_per_print": 10,
    "wall_clock_breakdown": True
}

with open("ds_config.json", "w") as f:
    json.dump(deepspeed_config, f)

image = image.add_local_file("ds_config.json", remote_path="/root/ds_config.json")

model_volume = modal.Volume.from_name("models", create_if_missing=True)

@app.function(
    image=image,
    timeout=300,
    secrets=[modal.Secret.from_name("HUGGING_FACE_TOKEN")],
    volumes = {"/model": model_volume}
)
def test_huggingface_auth(model_name=MODEL_NAME):
    """Test Hugging Face authentication and model access."""
    import os
    from huggingface_hub import login, whoami, model_info
    
    # Get token and login - ensuring consistent naming
    hf_token = os.environ.get("HUGGING_FACE_TOKEN")
    if not hf_token:
        return {"error": "HUGGING_FACE_TOKEN environment variable is not set"}
    
    login(token=hf_token)
    
    # Manually create token file to ensure consistent location
    os.makedirs(HF_TOKEN_DIR, exist_ok=True)
    with open(HF_TOKEN_PATH, 'w') as f:
        f.write(hf_token)
    
    # Set up cache directories
    model_cache_dir = "/model/transformers_cache"
    os.makedirs(model_cache_dir, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = model_cache_dir
    os.environ["HF_HOME"] = "/model/huggingface_home"
    os.environ["HF_DATASETS_CACHE"] = "/model/datasets_cache"
    
    # Test token file
    token_exists = os.path.exists(HF_TOKEN_PATH)
    print(f"Token file exists: {token_exists} at {HF_TOKEN_PATH}")
    
    # Test user info
    try:
        user = whoami()
        print(f"Logged in as: {user}")
    except Exception as e:
        return {"error": f"Failed to get user info: {str(e)}"}
    
    # Test model access
    try:
        info = model_info(model_name, token=hf_token)
        return {
            "success": True,
            "user": user,
            "model_access": True,
            "model_info": {
                "id": info.id,
                "private": info.private,
                "gated": info.gated
            },
            "token_file_exists": token_exists
        }
    except Exception as e:
        return {
            "success": True,
            "user": user,
            "model_access": False,
            "error": str(e),
            "token_file_exists": token_exists
        }

# --- Tests ---

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
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers import Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from huggingface_hub import login
    
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
def test_gpu_setup():
    """Test GPU configuration and multi-GPU support with all 8 GPUs."""
    import torch
    import os
    import subprocess
    import time
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    
    # Get number of GPUs
    num_gpus = torch.cuda.device_count() if cuda_available else 0
    
    # Get GPU info
    gpu_info = []
    total_memory_gb = 0
    if cuda_available:
        for i in range(num_gpus):
            memory_total = torch.cuda.get_device_properties(i).total_memory
            memory_total_gb = memory_total / (1024**3)  # Convert to GB
            total_memory_gb += memory_total_gb
            
            gpu_info.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_gb": f"{memory_total_gb:.2f}",
                "memory_allocated_mb": f"{torch.cuda.memory_allocated(i) / (1024**2):.2f}",
                "memory_reserved_mb": f"{torch.cuda.memory_reserved(i) / (1024**2):.2f}"
            })
    
    # Test multi-GPU compatibility
    distributed_available = hasattr(torch, 'distributed') and torch.distributed.is_available()
    
    # Test DeepSpeed
    try:
        import deepspeed
        deepspeed_available = True
        
        # Try initializing DeepSpeed
        from deepspeed.launcher.runner import main
        deepspeed_init_ok = True
    except Exception as e:
        deepspeed_available = False
        deepspeed_init_ok = False
    
    # Test NCCL for multi-GPU communication
    nccl_available = False
    if distributed_available:
        try:
            # Try to initialize process group with NCCL backend
            if torch.distributed.is_available() and not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl", init_method="tcp://localhost:12345", rank=0, world_size=1)
                nccl_available = True
                torch.distributed.destroy_process_group()
        except Exception as e:
            nccl_available = False
    
    # Bandwidth test between GPUs (simple implementation)
    gpu_bandwidth_results = []
    if num_gpus > 1:
        try:
            for src in range(min(3, num_gpus)):  # Test up to first 3 GPUs to save time
                for dst in range(min(3, num_gpus)):
                    if src != dst:
                        # Create tensors on source and destination
                        tensor_size = 1024 * 1024 * 100  # 100MB tensor
                        src_tensor = torch.ones(tensor_size, device=f"cuda:{src}")
                        dst_tensor = torch.zeros(tensor_size, device=f"cuda:{dst}")
                        
                        # Synchronize before timing
                        torch.cuda.synchronize(src)
                        torch.cuda.synchronize(dst)
                        
                        # Time the copy
                        start = time.time()
                        dst_tensor.copy_(src_tensor)
                        torch.cuda.synchronize(dst)
                        end = time.time()
                        
                        # Calculate bandwidth in GB/s
                        bandwidth = tensor_size * 4 / (1024**3) / (end - start)  # 4 bytes per float32
                        gpu_bandwidth_results.append({
                            "src": src,
                            "dst": dst,
                            "bandwidth_GBps": f"{bandwidth:.2f}"
                        })
        except Exception as e:
            gpu_bandwidth_results = [{"error": str(e)}]
    
    # Check if model can be loaded across multiple GPUs
    model_distribution_test = None
    if num_gpus >= 2:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Use tokenizer and model directly instead of config
            tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
            model = AutoModelForCausalLM.from_pretrained(
                "gpt2-medium",
                device_map="auto",
                torch_dtype=torch.float16  # Use half precision to save memory
            )
            
            # Get device mapping
            device_map = {}
            for name, param in model.named_parameters():
                device = param.device
                if device not in device_map:
                    device_map[str(device)] = 0
                device_map[str(device)] += param.numel() * param.element_size()
            
            # Convert to MB for readability
            for device in device_map:
                device_map[device] = f"{device_map[device] / (1024**2):.2f} MB"
            
            model_distribution_test = {
                "success": True,
                "device_map": device_map
            }
        except Exception as e:
            model_distribution_test = {
                "success": False,
                "error": str(e)
            }
    
    return {
        "cuda_available": cuda_available,
        "num_gpus": num_gpus,
        "total_gpu_memory_gb": f"{total_memory_gb:.2f}",
        "gpu_info": gpu_info,
        "distributed_available": distributed_available,
        "nccl_available": nccl_available,
        "deepspeed_available": deepspeed_available,
        "deepspeed_init_ok": deepspeed_init_ok,
        "gpu_bandwidth_tests": gpu_bandwidth_results,
        "model_distribution_test": model_distribution_test,
        "pytorch_version": torch.__version__
    }

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
def main(model_name=MODEL_NAME, test_auth=False, test_gpus=False):  # Set test_gpus default to False
    """Main entry point for the script."""
    # Load token from environment (only for local testing)
    from dotenv import load_dotenv
    load_dotenv()
    
    if test_auth:
        print("Testing Hugging Face authentication...")
        result = test_huggingface_auth.remote(model_name)
        print(json.dumps(result, indent=2))
        return
    
    # Always run GPU test before proceeding with training
    print("Testing GPU setup...")
    gpu_result = test_gpu_setup.remote()
    print(json.dumps(gpu_result, indent=2))
    
    # Check if we have enough GPUs for multi-GPU training
    if not gpu_result.get("cuda_available"):
        print("ERROR: CUDA is not available. Cannot proceed with GPU training.")
        return
    
    num_gpus = gpu_result.get("num_gpus", 0)
    if num_gpus < 2:
        print(f"WARNING: Only {num_gpus} GPU detected. Multi-GPU training requires at least 2 GPUs.")
        proceed = input("Do you want to proceed with single-GPU training? (y/n): ")
        if proceed.lower() != 'y':
            print("Training aborted.")
            return
    else:
        print(f"Found {num_gpus} GPUs. Proceeding with multi-GPU training.")
    
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
