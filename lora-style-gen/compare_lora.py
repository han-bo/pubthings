import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
LORA_PATH = "./output_lora"

def load_base_model():
    """加载基础模型（不带 LoRA）"""
    print("🔧 加载基础模型（不带 LoRA）...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

def load_lora_model():
    """加载带 LoRA 的模型"""
    print("🔧 加载基础模型...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("🔧 加载 LoRA 适配器...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    return model, tokenizer

def generate_text(model, tokenizer, features, max_new_tokens=180, temperature=0.7, top_p=0.9):
    """生成文案"""
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
    
    # 只返回生成的部分（去掉 prompt）
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取生成的部分
    if "文案：" in full_text:
        generated = full_text.split("文案：")[-1].strip()
    else:
        generated = full_text[len(prompt):].strip()
    
    return generated

def compare_single(features, base_model, base_tokenizer, lora_model, lora_tokenizer):
    """对比单个输入的生成结果"""
    print(f"\n{'='*80}")
    print(f"📦 商品特征: {features}")
    print(f"{'='*80}")
    
    # 基础模型生成
    print("\n🔵 【基础模型（无 LoRA）】")
    base_result = generate_text(base_model, base_tokenizer, features)
    print(base_result)
    
    # LoRA 模型生成
    print("\n🟢 【LoRA 微调模型】")
    lora_result = generate_text(lora_model, lora_tokenizer, features)
    print(lora_result)
    
    # 简单对比分析
    print(f"\n📊 对比分析:")
    print(f"  基础模型长度: {len(base_result)} 字符")
    print(f"  LoRA 模型长度: {len(lora_result)} 字符")
    print(f"  是否包含 emoji: 基础={('🔥' in base_result or '✨' in base_result or '💗' in base_result)}, LoRA={('🔥' in lora_result or '✨' in lora_result or '💗' in lora_result)}")
    
    return base_result, lora_result

def batch_compare(test_file="test_prompts.json", num_samples=5):
    """批量对比测试"""
    print("="*80)
    print("🚀 LoRA 微调效果对比测试")
    print("="*80)
    
    # 加载模型
    base_model, base_tokenizer = load_base_model()
    lora_model, lora_tokenizer = load_lora_model()
    
    # 读取测试用例
    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    
    # 限制测试数量
    test_cases = test_cases[:num_samples]
    
    print(f"\n📝 将测试 {len(test_cases)} 个用例\n")
    
    results = []
    for i, features in enumerate(test_cases, 1):
        print(f"\n【测试 {i}/{len(test_cases)}】")
        base_result, lora_result = compare_single(
            features, base_model, base_tokenizer, lora_model, lora_tokenizer
        )
        results.append({
            "features": features,
            "base": base_result,
            "lora": lora_result
        })
    
    # 总结
    print("\n" + "="*80)
    print("📈 测试总结")
    print("="*80)
    
    base_avg_len = sum(len(r["base"]) for r in results) / len(results)
    lora_avg_len = sum(len(r["lora"]) for r in results) / len(results)
    
    base_emoji_count = sum(1 for r in results if any(emoji in r["base"] for emoji in ["🔥", "✨", "💗", "💎", "🛏", "❄️"]))
    lora_emoji_count = sum(1 for r in results if any(emoji in r["lora"] for emoji in ["🔥", "✨", "💗", "💎", "🛏", "❄️"]))
    
    print(f"\n平均文案长度:")
    print(f"  基础模型: {base_avg_len:.1f} 字符")
    print(f"  LoRA 模型: {lora_avg_len:.1f} 字符")
    print(f"  差异: {lora_avg_len - base_avg_len:+.1f} 字符")
    
    print(f"\n包含 emoji 的样本数:")
    print(f"  基础模型: {base_emoji_count}/{len(results)} ({base_emoji_count/len(results)*100:.1f}%)")
    print(f"  LoRA 模型: {lora_emoji_count}/{len(results)} ({lora_emoji_count/len(results)*100:.1f}%)")
    
    return results

def interactive_compare():
    """交互式对比测试"""
    print("="*80)
    print("🚀 LoRA 微调效果对比测试（交互模式）")
    print("="*80)
    
    # 加载模型
    base_model, base_tokenizer = load_base_model()
    lora_model, lora_tokenizer = load_lora_model()
    
    print("\n✅ 模型加载完成！可以开始对比测试了。")
    print("💡 提示：直接回车使用默认测试用例，或输入自定义商品特征")
    
    while True:
        features = input("\n📦 输入商品特征（回车退出）: ").strip()
        if not features:
            break
        
        compare_single(features, base_model, base_tokenizer, lora_model, lora_tokenizer)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        # 交互模式
        interactive_compare()
    else:
        # 批量测试模式
        num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        batch_compare(num_samples=num_samples)

