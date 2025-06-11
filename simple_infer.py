import os
import argparse
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.prompting_utils import UniversalPrompting
from models.modeling_geomagvit import GeoMAGVIT
from peft import PeftModel
from torchvision import transforms

def find_bounds(image):
    np_image = np.array(image)
    non_white_pixels = np.any(np_image < [250, 250, 250], axis=-1)
    rows, cols = np.where(non_white_pixels)
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    return min_row, max_row, min_col, max_col


def crop(image, buffer: int = 20):
    min_row, max_row, min_col, max_col = find_bounds(image)
    min_row = max(0, min_row - buffer)
    max_row = min(image.height, max_row + buffer)
    min_col = max(0, min_col - buffer)
    max_col = min(image.width, max_col + buffer)
    return image.crop((min_col, min_row, max_col, max_row))


def expand2square(pil_img: Image.Image, background_color=(255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
    return result


def image_transform(image: Image.Image, resolution: int = 512):
    preprocess = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return preprocess(image)


def load_model(llm_path: str, adapter_path: str, device: torch.device):
    """Load base GeoUni LLM."""
    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    prompting = UniversalPrompting(
        tokenizer,
        max_len=4096,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|t2i|>", "<|mmu|>", "<|mix|>",
            "<formalization>", "</formalization>", "<answer>", "</answer>",
        ),
        ignore_id=-100,
    )
    
    # Attach reasoning adapter (LoRA) – only for MMU
    model = PeftModel.from_pretrained(model, adapter_path).to(device)
    model.eval()
    
    return model, tokenizer, prompting


def load_vq_model(vq_model_dir: str, device: torch.device):
    vq_model = GeoMAGVIT.from_pretrained(vq_model_dir, low_cpu_mem_usage=False).to(device)
    vq_model.eval().requires_grad_(False)
    return vq_model


def run_mixing(model, prompting, vq_model, prompt: str, save_path: str, device: torch.device):
    input_ids, _ = prompting(prompt, "mix_gen")
    input_ids = input_ids.to(device)

    with model.disable_adapter():
        image_tokens, text_tokens = model.mix_generate(
            input_ids=input_ids,
            max_new_tokens=2000,
            pad_token_id=prompting.text_tokenizer.pad_token_id,
            eos_token_id=prompting.text_tokenizer.eos_token_id,
            soi_token_id=prompting.text_tokenizer.convert_tokens_to_ids("<|soi|>"),
            eoi_token_id=prompting.text_tokenizer.convert_tokens_to_ids("<|eoi|>"),
            temperature=1.0,
        )

    # decode image
    image = vq_model.decode_code(image_tokens)
    image = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0) * 255.0
    image = image[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    Image.fromarray(image).save(os.path.join(save_path, "geouni_mixing_sample.png"))

    response = prompting.text_tokenizer.batch_decode(text_tokens, skip_special_tokens=True)[0]
    print("[Mixing] Response:\n", response)
    print("[Mixing] Diagram:", os.path.join(save_path, "geouni_mixing_sample.png"))


def run_t2d(model, prompting, vq_model, prompt: str, save_path: str, device: torch.device):
    input_ids, attention_masks = prompting(prompt, "t2i_gen")
    input_ids, attention_masks = input_ids.to(device), attention_masks.to(device)

    
    with model.disable_adapter():
        code_ids = model.t2i_generate(
            input_ids=input_ids,
            attention_masks=attention_masks,
            pad_token_id=prompting.text_tokenizer.pad_token_id,
            temperature=1.0,
        )

    image = vq_model.decode_code(code_ids)
    image = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0) * 255.0
    image = image[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    Image.fromarray(image).save(os.path.join(save_path, "geouni_t2i_sample.png"))

    print("[T2D] Image saved to", os.path.join(save_path, "geouni_t2i_sample.png"))


def run_mmu(model, prompting, vq_model, image_path: str, question: str, device: torch.device):
    

    # Prepare image tokens
    img = Image.open(image_path).convert("RGB")
    img = crop(img)
    img = expand2square(img, (255, 255, 255))
    img_tensor = image_transform(img, resolution=512).unsqueeze(0).to(device)
    image_tokens = vq_model.get_code(img_tensor)

    prompt = f"Analyze the input geometry image to extract consCDL and imgCDL, then answer the question.\nQuestion: {question}"
    input_ids, _ = prompting([image_tokens, prompt], "mmu_gen")

    with torch.no_grad():
        output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=2000,
                    temperature=1.0,
                    pad_token_id=prompting.text_tokenizer.pad_token_id,
                    eos_token_id=prompting.text_tokenizer.eos_token_id,
                    do_sample=False,
                    top_p=None,
                    use_cache=True,
                )
        response = prompting.text_tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        print("[MMU] Response:\n", response)


def main():
    parser = argparse.ArgumentParser(description="Unified inference script for GeoUni tasks (mixing, t2i, mmu)")
    parser.add_argument("mode", choices=["mixing", "t2d", "mmu"], help="Select inference mode")
    parser.add_argument("--save_dir", default="./outputs", help="Directory to save generated images")
    parser.add_argument("--prompt", default=None, help="Text prompt for mixing or t2i mode")
    parser.add_argument("--image_path", default=None, help="Input image path for MMU mode")
    parser.add_argument("--question", default="", help="Question sentence for MMU mode")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    llm_path = "JO-KU/GeoUni-Instruct"
    adapter_path = "JO-KU/GeoUni-Reasoning-Adapter"
    vq_path = "JO-KU/Geo-MAGVIT"
    
    model, tokenizer, prompting = load_model(llm_path, adapter_path, device)
    vq_model = load_vq_model(vq_path, device)

    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == "mixing":
        if args.prompt is None:
            args.prompt = (
                "Draw a diagram, create a question and answer based on the given knowledge point. "
                "Knowledge point: definition of a midpoint, definition of a median of a triangle, "
                "properties of a median, algebraic operations."
            )
        run_mixing(model, prompting, vq_model, args.prompt, args.save_dir, device)

    elif args.mode == "t2d":
        if args.prompt is None:
            args.prompt = (
                "Draw a geometric image based on this description: The diagram involves a circle centered "
                "at O with points A, B, C, D, and E all lying on it. Given values include ∠CAB=25° and "
                "∠DEC=30°. Points AFMC, BMO, DNO, ENC, and BFE are collinear, indicating certain lines "
                "within the figure."
            )
        run_t2d(model, prompting, vq_model, args.prompt, args.save_dir, device)

    elif args.mode == "mmu":
        if args.image_path is None:
            raise ValueError("--image_path is required for mmu mode")
        if not os.path.isfile(args.image_path):
            raise FileNotFoundError(args.image_path)
        run_mmu(model, prompting, vq_model, args.image_path, args.question, device)


if __name__ == "__main__":
    main()
