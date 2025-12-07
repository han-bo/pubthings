# PubThings 🚀

> **开源 AI 项目集合 | 从 LoRA 微调到 RAG 应用，探索大语言模型的实践之路**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-orange.svg)](https://www.langchain.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

一个专注于 **大语言模型应用实践** 的开源项目集合，包含从模型微调到 RAG 应用的完整解决方案。所有项目都经过实际验证，代码清晰易懂，适合学习和生产使用。

## ✨ 为什么选择 PubThings？

- 🎯 **实用性强**：所有项目都经过实际验证，可直接用于生产环境
- 📚 **学习友好**：代码注释详细，包含完整的工作流程说明
- 🔒 **隐私保护**：支持完全本地化部署，数据不出本地
- 🛠️ **开箱即用**：提供详细的文档和快速开始指南
- 🌟 **持续更新**：项目持续维护，紧跟最新技术发展

## 🎁 项目亮点

### 🎨 LoRA 微调实战
- 使用 **LoRA** 技术高效微调大模型
- 完整的小红书风格文案生成解决方案
- 包含训练、推理、评估全流程

### 📖 RAG 应用实践
- 基于 **LangChain + ChromaDB** 的文档问答系统
- 支持多种文档格式（Word、PDF、Markdown 等）
- 完整的日志记录和调试功能

---

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

### 🎨 LoRA 风格生成器

3 步开始你的第一个 LoRA 微调项目：

```bash
cd lora-style-gen
pip install torch transformers peft accelerate gradio
python train_lora.py  # 训练模型
python gradio_app.py   # 启动 Web 界面
```

**适用场景：** 文本风格迁移、领域适配、个性化生成

### 📖 Mini-RAG 文档问答

5 分钟搭建你的本地知识库：

```bash
cd min-rag
pip install -r requirements.txt
ollama pull qwen2.5:1.5b  # 下载模型（约 1GB）
ollama serve              # 启动 Ollama 服务
python app.py             # 运行 RAG 系统
```

**适用场景：** 企业内部知识库、文档问答、智能客服

## 📊 项目统计

- 📦 **2** 个完整项目
- 💻 **1000+** 行代码
- 📚 **详细文档** 和注释
- 🔧 **开箱即用** 的配置

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

如果你觉得这个项目有帮助，请给一个 ⭐ Star，这是对我们最大的支持！

## 📚 学习资源

每个项目都包含：
- 📖 详细的 README 文档
- 💡 代码注释和工作流程说明
- 🔍 日志记录和调试功能
- 📝 最佳实践建议

## 🌟 Star History

如果这个项目对你有帮助，请考虑：

- ⭐ **Star** 这个仓库
- 🍴 **Fork** 并创建你的版本
- 🐛 **报告 Bug** 或提出改进建议
- 📖 **分享** 给其他开发者

## 📄 许可证

请参考各个子项目的许可证说明。

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给一个 Star！⭐**

Made with ❤️ by [PubThings Contributors](https://github.com/your-username/pubthings/graphs/contributors)

</div>
