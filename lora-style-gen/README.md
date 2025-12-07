# 🌸 小红书风格文案生成器 - LoRA 微调项目

基于 Qwen2-0.5B-Instruct 模型，使用 LoRA 技术微调生成小红书风格的文案。

## 📋 项目简介

本项目通过 LoRA (Low-Rank Adaptation) 技术对 Qwen2-0.5B-Instruct 模型进行微调，使其能够生成符合小红书平台风格的文案。项目包含完整的训练、推理、评估和 Web 界面功能。

## ✨ 主要特性

- 🎯 **LoRA 微调**：使用 PEFT 库实现高效的参数高效微调
- 📝 **风格生成**：生成符合小红书风格的文案（包含 emoji、感叹词、情感表达等）
- 🔍 **效果对比**：提供基础模型与微调模型的详细对比分析
- 🌐 **Web 界面**：基于 Gradio 的交互式 Web 应用
- 📊 **风格评估**：自动评估生成文案的小红书风格得分

## 🛠️ 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA（推荐，用于 GPU 加速）

## 📦 安装依赖

```bash
pip install torch transformers peft accelerate gradio
```

## 📁 项目结构

```
lora-style-gen/
├── train_lora.py          # LoRA 训练脚本
├── inference_lora.py      # 推理脚本（交互式）
├── inference.py           # 推理脚本（简化版）
├── gradio_app.py          # Gradio Web 界面
├── compare_lora.py        # 模型效果对比脚本
├── train.json             # 训练数据（JSON 格式）
├── xiaohongshu_200.jsonl  # 训练数据（JSONL 格式，200 条）
├── test_prompts.json      # 测试用例
└── output_lora/           # LoRA 模型输出目录
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── checkpoint-*/      # 训练检查点
```

## 🚀 快速开始

### 1. 准备训练数据

训练数据格式为 JSONL，每行包含：
```json
{
  "instruction": "请根据商品特征写一个小红书风格的文案：",
  "input": "蓝牙耳机，续航长",
  "output": "真的绝了！这个蓝牙耳机续航超长✨ 入手不亏～"
}
```

### 2. 训练 LoRA 模型

```bash
python train_lora.py
```

训练参数（可在 `train_lora.py` 中修改）：
- **基础模型**：`Qwen/Qwen2-0.5B-Instruct`
- **LoRA 配置**：
  - `r=6`：LoRA 秩
  - `lora_alpha=8`：LoRA 缩放参数
  - `target_modules=["q_proj", "v_proj"]`：目标模块
- **训练参数**：
  - `max_steps=200`：训练步数
  - `learning_rate=4e-5`：学习率
  - `per_device_train_batch_size=1`：批次大小
  - `gradient_accumulation_steps=4`：梯度累积步数

训练完成后，模型将保存到 `output_lora/` 目录。

### 3. 使用模型生成文案

#### 方式一：交互式推理

```bash
python inference_lora.py
```

#### 方式二：使用 Gradio Web 界面

```bash
python gradio_app.py
```

然后在浏览器中访问 `http://localhost:7860`

#### 方式三：在代码中使用

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
LORA_PATH = "./output_lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

# 生成文案
features = "蓝牙耳机，续航长"
prompt = f"请根据商品特征写一个小红书风格的文案：\n商品：{features}\n文案："
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=180,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

### 4. 对比模型效果

#### 批量对比测试

```bash
python compare_lora.py 5  # 测试 5 个用例
```

#### 交互式对比

```bash
python compare_lora.py --interactive
```

对比脚本会评估：
- 📊 小红书风格评分（0-100 分）
- ✨ Emoji 使用情况
- 💬 感叹词和情感词使用
- 📏 文案长度
- 📚 与训练数据的相似度

## 📊 模型配置

### LoRA 参数

当前配置（可在 `train_lora.py` 中修改）：

```python
LoraConfig(
    r=6,                    # LoRA 秩，控制参数量
    lora_alpha=8,          # LoRA 缩放参数
    target_modules=["q_proj", "v_proj"],  # 目标注意力模块
    lora_dropout=0.05,      # Dropout 率
    bias="none",            # 不训练偏置
    task_type="CAUSAL_LM",  # 因果语言模型
)
```

### 训练参数

```python
TrainingArguments(
    output_dir="output_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=200,
    learning_rate=4e-5,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
)
```

## 🎨 小红书风格特征

模型学习的小红书风格特征包括：

- **Emoji 使用**：🔥 ✨ 💗 💎 等表情符号
- **感叹词**：真的、太、超、绝了、爱了、必入等
- **情感表达**：治愈、幸福、舒服、上头等
- **结尾标记**：～、！、！～ 等
- **口语化表达**：入手、不亏、救星、神器等
- **文案长度**：通常在 25-60 字符之间

## 📈 效果评估

`compare_lora.py` 脚本提供了详细的评估指标：

1. **风格评分**：基于多个维度计算 0-100 分的风格得分
2. **特征覆盖率**：统计各种风格特征的出现频率
3. **与训练数据相似度**：计算生成文本与训练样本的相似度
4. **综合评估**：给出微调效果的总体评价和改进建议

## 🔧 自定义配置

### 修改基础模型

在 `train_lora.py` 中修改：

```python
base_model = "Qwen/Qwen2-0.5B-Instruct"  # 改为其他模型
```

### 调整 LoRA 参数

增加 `r` 值可以提升模型容量，但会增加参数量和训练时间：

```python
lora_config = LoraConfig(
    r=16,  # 从 6 增加到 16
    lora_alpha=32,  # 通常设为 r 的 2 倍
    # ...
)
```

### 修改训练数据

1. 准备 JSONL 格式的数据文件
2. 在 `train_lora.py` 中修改 `train_file` 路径
3. 根据需要调整 `max_steps` 等训练参数

## 📝 数据格式

### 训练数据格式（JSONL）

每行一个 JSON 对象：

```json
{"instruction": "请根据商品特征写一个小红书风格的文案：", "input": "商品特征描述", "output": "小红书风格文案"}
```

### 测试数据格式（JSON）

```json
[
  "商品特征1",
  "商品特征2",
  ...
]
```

## 🐛 常见问题

### 1. 内存不足

- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps` 保持有效批次大小
- 使用 `fp16=True` 启用混合精度训练

### 2. 生成效果不佳

- 增加训练数据量（建议至少 100+ 条）
- 增加训练步数（`max_steps`）
- 调整 LoRA 参数（增加 `r` 值）
- 检查训练数据质量

### 3. 模型加载失败

- 确保 `output_lora/` 目录存在且包含 `adapter_config.json` 和 `adapter_model.safetensors`
- 检查基础模型路径是否正确

## 📚 相关资源

- [PEFT 文档](https://huggingface.co/docs/peft)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [Qwen 模型](https://huggingface.co/Qwen)

## 📄 许可证

请参考基础模型 Qwen2-0.5B-Instruct 的许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请通过 Issue 反馈。

