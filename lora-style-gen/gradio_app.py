import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

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

def generate_xhs_text(features, temperature, max_tokens):
    prompt = f"请根据商品特征写一段小红书风格的文案：\n商品：{features}\n文案："
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

with gr.Blocks() as demo:
    gr.Markdown("# 🌸 小红书风格文案生成器（LoRA 微调版）")

    with gr.Row():
        features = gr.Textbox(label="商品特征", placeholder="例：蓝牙耳机，续航长，佩戴舒适")
    
    with gr.Row():
        temperature = gr.Slider(0.2, 1.5, 0.7, label="Temperature")
        max_tokens = gr.Slider(50, 300, 150, label="Max New Tokens")

    result = gr.Textbox(label="生成文案")

    submit_btn = gr.Button("生成文案 ✨")
    submit_btn.click(
        fn=generate_xhs_text,
        inputs=[features, temperature, max_tokens],
        outputs=result
    )

demo.launch(server_name="0.0.0.0", server_port=7860)

