import torch
import argparse
from hi_diffusers import HiDreamImagePipeline
from hi_diffusers import HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="dev")
parser.add_argument("--use_quantize_model", action="store_true", help="Whether to use the quantized model")   
parser.add_argument("--cpu_offload", action="store_true", help="Whether to enable cpu_offload")   
args = parser.parse_args()
model_type = args.model_type
MODEL_PREFIX = "HiDream-ai"
if args.use_quantize_model:
    LLAMA_MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
else:
    LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Model configurations
MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    }
}

# Resolution options
RESOLUTION_OPTIONS = [
    "1024 × 1024 (Square)",
    "768 × 1360 (Portrait)",
    "1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)",
    "1168 × 880 (Landscape)",
    "1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]

# Load models
def load_models(model_type):
    config = MODEL_CONFIGS[model_type]
    pretrained_model_name_or_path = config["path"]
    if args.use_quantize_model:
        pretrained_model_name_or_path = f"{pretrained_model_name_or_path}-nf4"
    scheduler = MODEL_CONFIGS[model_type]["scheduler"](num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)
    
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
        LLAMA_MODEL_NAME,
        use_fast=False)
    
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        output_hidden_states=True,
        output_attentions=True,
        device_map="auto",
        torch_dtype=torch.bfloat16)

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="transformer", 
        device_map="auto",
        torch_dtype=torch.bfloat16)  
    
    pipe = HiDreamImagePipeline.from_pretrained(
        pretrained_model_name_or_path, 
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16
    )
    pipe.transformer = transformer
    pipe.text_encoder.to(torch.bfloat16)
    pipe.text_encoder_2.to(torch.bfloat16)
    if args.cpu_offload:
        pipe.enable_sequential_cpu_offload()    
    else:
        pipe.to("cuda") 
    return pipe, config

# Parse resolution string to get height and width
def parse_resolution(resolution_str):
    if "1024 × 1024" in resolution_str:
        return 1024, 1024
    elif "768 × 1360" in resolution_str:
        return 768, 1360
    elif "1360 × 768" in resolution_str:
        return 1360, 768
    elif "880 × 1168" in resolution_str:
        return 880, 1168
    elif "1168 × 880" in resolution_str:
        return 1168, 880
    elif "1248 × 832" in resolution_str:
        return 1248, 832
    elif "832 × 1248" in resolution_str:
        return 832, 1248
    else:
        return 1024, 1024  # Default fallback

# Generate image function
def generate_image(pipe, model_type, prompt, resolution, seed):
    # Get configuration for current model
    config = MODEL_CONFIGS[model_type]
    guidance_scale = config["guidance_scale"]
    num_inference_steps = config["num_inference_steps"]
    
    # Parse resolution
    height, width = parse_resolution(resolution)
    
    # Handle seed
    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    images = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,
        generator=generator
    ).images
    
    return images[0], seed

# Initialize with default model
print(f"Loading default model ({model_type})...")
pipe, _ = load_models(model_type)
print("Model loaded successfully!")
prompt = "A cat holding a sign that says \"Hi-Dreams.ai\"." 
resolution = "1024 × 1024 (Square)"
seed = -1
image, seed = generate_image(pipe, model_type, prompt, resolution, seed)
image.save("output.png")
