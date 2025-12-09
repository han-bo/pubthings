import json
import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from peft import LoraConfig, get_peft_model


# =====================
# 1. 准备数据集
# =====================

class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                instruction = obj["instruction"]
                _input = obj["input"]
                output = obj["output"]

                # 构建 prompt 格式：你可以根据模型格式调整
                prompt = f"{instruction}\n输入：{_input}\n输出："
                target = output

                self.samples.append((prompt, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, target = self.samples[idx]
        
        # 模型训练格式为 prompt + answer
        full_text = prompt + target
        
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
        )
        return {
            "input_ids": torch.tensor(tokenized["input_ids"]),
            "attention_mask": torch.tensor(tokenized["attention_mask"]),
            "labels": torch.tensor(tokenized["input_ids"]),  # causal LM 直接预测下一个 token
        }


# =====================
# 2. 加载模型与 LoRA
# =====================

def load_model_and_tokenizer(base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # LoRA 配置
    lora_config = LoraConfig(
        r=6,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],   # 对 Qwen 非常适用
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# =====================
# 3. 开始训练
# =====================

def train():
    base_model = "Qwen/Qwen2-0.5B-Instruct"
    #train_file = "train.json"
    train_file = "xiaohongshu_200.jsonl"
    output_dir = "output_lora"

    model, tokenizer = load_model_and_tokenizer(base_model)

    dataset = JsonlDataset(train_file, tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=200,   # 少量步骤即可跑通
        learning_rate=4e-5,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        optim="adamw_torch",
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # 保存 LoRA adapter
    model.save_pretrained(output_dir)
    print("🎉 LoRA 训练完成！权重已保存到 output_lora/")


if __name__ == "__main__":
    train()

