# PubThings

存放代码、文章等项目资源。

## 📁 项目目录

### [lora-style-gen](./lora-style-gen/)

小红书风格文案生成器 - 基于 Qwen2-0.5B-Instruct 的 LoRA 微调项目。

**主要功能：**
- 🎯 LoRA 微调训练
- 📝 小红书风格文案生成
- 🔍 模型效果对比评估
- 🌐 Gradio Web 界面

**技术栈：**
- Python 3.8+
- PyTorch 2.0+
- PEFT (LoRA)
- Transformers
- Gradio

详细文档请查看 [lora-style-gen/README.md](./lora-style-gen/README.md)

---

### [min-rag](./min-rag/)

Mini-RAG：本地文档问答系统 - 基于 LangChain + ChromaDB + Ollama 的轻量级 RAG 项目。

**主要功能：**
- 📄 自动加载本地文档（支持 .docx、.doc、.txt、.md 等格式）
- ✂️ 文本切分 + 向量化
- 🗄️ Chroma 本地向量数据库
- 🤖 本地 LLM（通过 Ollama）回答问题
- 📝 完整的对话日志记录
- 🔍 检索增强生成（RAG）全流程

**技术栈：**
- Python 3.9+
- LangChain
- ChromaDB
- Sentence-Transformers
- Ollama + Qwen/DeepSeek/Llama

**特色：**
- ✅ 完全本地化部署，保护数据隐私
- ✅ 支持多种文档格式
- ✅ 详细的日志记录（问题、检索内容、Prompt、回答）
- ✅ 轻量级设计，适合学习和生产使用

详细文档请查看 [min-rag/README.md](./min-rag/README.md)

---

## 🚀 快速开始

### LoRA 风格生成器

```bash
cd lora-style-gen
pip install torch transformers peft accelerate gradio
python train_lora.py  # 训练模型
python gradio_app.py   # 启动 Web 界面
```

### Mini-RAG 文档问答

```bash
cd min-rag
pip install -r requirements.txt
ollama pull qwen2.5:1.5b  # 下载模型
ollama serve              # 启动 Ollama 服务
python app.py             # 运行 RAG 系统
```

## 📝 说明

本仓库用于存放各种代码项目和文章资源。每个子项目都有独立的 README 文档，包含详细的安装和使用说明。

## 📄 许可证

请参考各个子项目的许可证说明。
