import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
LORA_PATH = "./output_lora"   # 你的 LoRA 权重路径

print("🔧 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print("🔧 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("🔧 Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()


def generate_xhs_text(features, max_new_tokens=180, temperature=0.7, top_p=0.9):
    """
    features: 商品特征，例如 "蓝牙耳机，续航长"
    """
    prompt = f"请根据商品特征写一个小红书风格的文案：\n商品：{features}\n文案："

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    print("=== 小红书 LoRA 模型测试 ===")
    while True:
        features = input("\n输入商品特征：")
        if not features.strip():
            continue
        print("\n📌 生成结果：")
        print(generate_xhs_text(features))

