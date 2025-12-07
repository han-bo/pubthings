import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from collections import Counter

BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
LORA_PATH = "./output_lora"

# 小红书风格特征关键词
XHS_STYLE_KEYWORDS = {
    "emoji": ["🔥", "✨", "💗", "💎", "🛏", "❄️", "🤖", "😱", "☀️", "💄", "👁️", "🍼", 
              "👶", "🥛", "🥜", "🍃", "🧳", "🌸", "🩴", "🛁", "💁‍♀️", "💅", "😴", 
              "💧", "🐱", "🐶", "⛺", "🌧️", "⌨️", "💻", "🎒", "💤", "🦷", "🌬", 
              "🍟", "🪑", "🫧", "🏃‍♂️", "🧘‍♀️"],
    "exclamations": ["真的", "太", "超", "绝了", "爱了", "必入", "不亏", "救星", "神器", 
                     "太绝了", "真的爱了", "真的绝", "太幸福", "太舒服", "太方便"],
    "emotional_words": ["治愈", "幸福", "舒服", "爱了", "绝了", "上头", "拉满", "必囤"],
    "ending_markers": ["～", "！", "！～", "～！"],
    "oral_expressions": ["入手", "必入", "不亏", "救星", "神器", "嘴馋", "手残党", "铲屎官"]
}

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

def evaluate_xhs_style(text):
    """评估文本的小红书风格得分（0-100分）"""
    score = 0
    features = {
        "has_emoji": False,
        "emoji_count": 0,
        "has_exclamation": False,
        "exclamation_count": 0,
        "has_emotional": False,
        "emotional_count": 0,
        "has_ending_marker": False,
        "has_oral": False,
        "oral_count": 0,
        "length_score": 0
    }
    
    # 1. Emoji 检测（30分）
    emoji_pattern = r'[\U0001F300-\U0001F9FF]|[\u2600-\u27FF]'
    emojis = re.findall(emoji_pattern, text)
    features["emoji_count"] = len(emojis)
    features["has_emoji"] = len(emojis) > 0
    if features["has_emoji"]:
        score += 30
        if features["emoji_count"] >= 2:
            score += 5  # 多个 emoji 加分
    
    # 2. 感叹词检测（25分）
    for word in XHS_STYLE_KEYWORDS["exclamations"]:
        if word in text:
            features["has_exclamation"] = True
            features["exclamation_count"] += text.count(word)
    if features["has_exclamation"]:
        score += 25
    
    # 3. 情感词汇检测（20分）
    for word in XHS_STYLE_KEYWORDS["emotional_words"]:
        if word in text:
            features["has_emotional"] = True
            features["emotional_count"] += text.count(word)
    if features["has_emotional"]:
        score += 20
    
    # 4. 结尾标记（10分）
    for marker in XHS_STYLE_KEYWORDS["ending_markers"]:
        if text.endswith(marker):
            features["has_ending_marker"] = True
            score += 10
            break
    
    # 5. 口语化表达（10分）
    for expr in XHS_STYLE_KEYWORDS["oral_expressions"]:
        if expr in text:
            features["has_oral"] = True
            features["oral_count"] += text.count(expr)
    if features["has_oral"]:
        score += 10
    
    # 6. 长度评分（5分）- 小红书文案通常在 25-60 字符
    length = len(text)
    if 25 <= length <= 60:
        features["length_score"] = 5
        score += 5
    elif 20 <= length < 25 or 60 < length <= 80:
        features["length_score"] = 3
        score += 3
    
    # 限制最高分
    score = min(score, 100)
    
    return score, features

def load_training_samples():
    """加载训练样本用于参考"""
    samples = []
    try:
        with open("train.json", "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                samples.append(obj["output"])
    except:
        pass
    return samples

def calculate_similarity_to_training(text, training_samples):
    """计算与训练样本的相似度（基于共同词汇）"""
    if not training_samples:
        return 0
    
    # 提取文本中的关键词（去除标点和常见词）
    def extract_keywords(t):
        # 移除 emoji 和标点
        t_clean = re.sub(r'[\U0001F300-\U0001F9FF]|[\u2600-\u27FF]', '', t)
        t_clean = re.sub(r'[，。！？～\s]', '', t_clean)
        return set(t_clean)
    
    text_keywords = extract_keywords(text)
    
    # 计算与所有训练样本的平均相似度
    similarities = []
    for sample in training_samples:
        sample_keywords = extract_keywords(sample)
        if len(text_keywords) == 0 or len(sample_keywords) == 0:
            continue
        intersection = len(text_keywords & sample_keywords)
        union = len(text_keywords | sample_keywords)
        if union > 0:
            similarities.append(intersection / union)
    
    return sum(similarities) / len(similarities) * 100 if similarities else 0

def compare_single(features, base_model, base_tokenizer, lora_model, lora_tokenizer, training_samples=None):
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
    
    # 详细评估分析
    print(f"\n📊 详细评估分析:")
    print(f"{'-'*80}")
    
    # 风格评分
    base_score, base_features = evaluate_xhs_style(base_result)
    lora_score, lora_features = evaluate_xhs_style(lora_result)
    
    print(f"\n🎯 小红书风格评分 (0-100分):")
    print(f"  基础模型: {base_score:.1f} 分")
    print(f"  LoRA 模型: {lora_score:.1f} 分")
    print(f"  提升: {lora_score - base_score:+.1f} 分")
    
    # 详细特征对比
    print(f"\n📝 风格特征对比:")
    print(f"  Emoji: 基础={base_features['emoji_count']}个, LoRA={lora_features['emoji_count']}个")
    print(f"  感叹词: 基础={base_features['exclamation_count']}个, LoRA={lora_features['exclamation_count']}个")
    print(f"  情感词: 基础={base_features['emotional_count']}个, LoRA={lora_features['emotional_count']}个")
    print(f"  口语化: 基础={base_features['oral_count']}个, LoRA={lora_features['oral_count']}个")
    print(f"  结尾标记: 基础={'✓' if base_features['has_ending_marker'] else '✗'}, LoRA={'✓' if lora_features['has_ending_marker'] else '✗'}")
    print(f"  长度: 基础={len(base_result)}字符, LoRA={len(lora_result)}字符")
    
    # 与训练数据相似度
    if training_samples:
        base_sim = calculate_similarity_to_training(base_result, training_samples)
        lora_sim = calculate_similarity_to_training(lora_result, training_samples)
        print(f"\n📚 与训练数据相似度:")
        print(f"  基础模型: {base_sim:.1f}%")
        print(f"  LoRA 模型: {lora_sim:.1f}%")
        print(f"  提升: {lora_sim - base_sim:+.1f}%")
    
    # 综合判断
    print(f"\n{'='*80}")
    if lora_score > base_score + 10:
        print("✅ LoRA 微调效果显著！风格更接近小红书文案")
    elif lora_score > base_score + 5:
        print("⚠️  LoRA 微调有一定效果，但仍有改进空间")
    elif lora_score > base_score:
        print("⚠️  LoRA 微调效果微弱，可能需要更多训练数据或调整参数")
    else:
        print("❌ LoRA 微调效果不明显，建议检查训练过程")
    print(f"{'='*80}")
    
    return {
        "base": base_result,
        "lora": lora_result,
        "base_score": base_score,
        "lora_score": lora_score,
        "base_features": base_features,
        "lora_features": lora_features
    }

def batch_compare(test_file="test_prompts.json", num_samples=5):
    """批量对比测试"""
    print("="*80)
    print("🚀 LoRA 微调效果对比测试")
    print("="*80)
    
    # 加载模型
    base_model, base_tokenizer = load_base_model()
    lora_model, lora_tokenizer = load_lora_model()
    
    # 加载训练样本用于参考
    training_samples = load_training_samples()
    if training_samples:
        print(f"📚 已加载 {len(training_samples)} 个训练样本用于参考")
    
    # 读取测试用例
    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    
    # 限制测试数量
    test_cases = test_cases[:num_samples]
    
    print(f"\n📝 将测试 {len(test_cases)} 个用例\n")
    
    results = []
    for i, features in enumerate(test_cases, 1):
        print(f"\n【测试 {i}/{len(test_cases)}】")
        result = compare_single(
            features, base_model, base_tokenizer, lora_model, lora_tokenizer, training_samples
        )
        results.append({
            "features": features,
            **result
        })
    
    # 总结
    print("\n" + "="*80)
    print("📈 综合评估总结")
    print("="*80)
    
    # 基础统计
    base_avg_len = sum(len(r["base"]) for r in results) / len(results)
    lora_avg_len = sum(len(r["lora"]) for r in results) / len(results)
    
    base_avg_score = sum(r["base_score"] for r in results) / len(results)
    lora_avg_score = sum(r["lora_score"] for r in results) / len(results)
    
    # Emoji 统计
    base_emoji_count = sum(1 for r in results if r["base_features"]["has_emoji"])
    lora_emoji_count = sum(1 for r in results if r["lora_features"]["has_emoji"])
    
    # 感叹词统计
    base_excl_count = sum(1 for r in results if r["base_features"]["has_exclamation"])
    lora_excl_count = sum(1 for r in results if r["lora_features"]["has_exclamation"])
    
    # 情感词统计
    base_emo_count = sum(1 for r in results if r["base_features"]["has_emotional"])
    lora_emo_count = sum(1 for r in results if r["lora_features"]["has_emotional"])
    
    # 结尾标记统计
    base_end_count = sum(1 for r in results if r["base_features"]["has_ending_marker"])
    lora_end_count = sum(1 for r in results if r["lora_features"]["has_ending_marker"])
    
    print(f"\n🎯 平均风格评分:")
    print(f"  基础模型: {base_avg_score:.1f} 分")
    print(f"  LoRA 模型: {lora_avg_score:.1f} 分")
    print(f"  平均提升: {lora_avg_score - base_avg_score:+.1f} 分")
    
    print(f"\n📏 平均文案长度:")
    print(f"  基础模型: {base_avg_len:.1f} 字符")
    print(f"  LoRA 模型: {lora_avg_len:.1f} 字符")
    print(f"  差异: {lora_avg_len - base_avg_len:+.1f} 字符")
    
    print(f"\n✨ 风格特征覆盖率:")
    print(f"  Emoji: 基础={base_emoji_count}/{len(results)} ({base_emoji_count/len(results)*100:.1f}%), LoRA={lora_emoji_count}/{len(results)} ({lora_emoji_count/len(results)*100:.1f}%)")
    print(f"  感叹词: 基础={base_excl_count}/{len(results)} ({base_excl_count/len(results)*100:.1f}%), LoRA={lora_excl_count}/{len(results)} ({lora_excl_count/len(results)*100:.1f}%)")
    print(f"  情感词: 基础={base_emo_count}/{len(results)} ({base_emo_count/len(results)*100:.1f}%), LoRA={lora_emo_count}/{len(results)} ({lora_emo_count/len(results)*100:.1f}%)")
    print(f"  结尾标记: 基础={base_end_count}/{len(results)} ({base_end_count/len(results)*100:.1f}%), LoRA={lora_end_count}/{len(results)} ({lora_end_count/len(results)*100:.1f}%)")
    
    # 最终评估
    print(f"\n{'='*80}")
    print("🎓 微调效果评估")
    print(f"{'='*80}")
    
    score_improvement = lora_avg_score - base_avg_score
    emoji_improvement = (lora_emoji_count - base_emoji_count) / len(results) * 100
    
    if score_improvement >= 15 and emoji_improvement >= 20:
        print("✅ 优秀！LoRA 微调效果非常显著")
        print("   - 风格评分大幅提升")
        print("   - 小红书特征明显增强")
        print("   - 建议：可以继续使用当前模型")
    elif score_improvement >= 10 and emoji_improvement >= 10:
        print("✅ 良好！LoRA 微调有明显效果")
        print("   - 风格评分有所提升")
        print("   - 部分特征得到改善")
        print("   - 建议：可以尝试增加训练步数或调整学习率")
    elif score_improvement >= 5:
        print("⚠️  一般！LoRA 微调效果有限")
        print("   - 风格评分略有提升")
        print("   - 特征改善不明显")
        print("   - 建议：增加训练数据、调整 LoRA 参数（r值）或增加训练步数")
    else:
        print("❌ 较差！LoRA 微调效果不明显")
        print("   - 风格评分提升很小或没有提升")
        print("   - 建议：")
        print("     1. 检查训练数据质量和数量（建议至少 100+ 样本）")
        print("     2. 增加训练步数（当前 200 步可能不够）")
        print("     3. 调整 LoRA 参数：r=16, alpha=32")
        print("     4. 检查训练损失是否正常下降")
    
    print(f"{'='*80}")
    
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
    
    # 加载训练样本用于参考
    training_samples = load_training_samples()
    
    while True:
        features = input("\n📦 输入商品特征（回车退出）: ").strip()
        if not features:
            break
        
        compare_single(features, base_model, base_tokenizer, lora_model, lora_tokenizer, training_samples)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        # 交互模式
        interactive_compare()
    else:
        # 批量测试模式
        num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        batch_compare(num_samples=num_samples)

