# Mini-RAG：本地文档问答系统

一个基于 Python + LangChain + ChromaDB + 本地大模型（Qwen/DeepSeek/Llama）的轻量级 RAG 项目。
用于快速学习 RAG 全流程。

## 🚀 功能
- 自动加载本地文档（支持 .docx、.doc、.txt、.md 等格式）
- 文本切分 + 向量化
- Chroma 本地向量数据库
- 本地 LLM（通过 Ollama）回答问题
- 输出答案与引用片段

## 🛠 技术栈
- Python 3.9+
- LangChain
- ChromaDB
- Sentence-Transformers
- Ollama + Qwen/DeepSeek/Llama

## 📋 前置要求

1. **安装 Ollama**
   ```bash
   # macOS
   brew install ollama
   
   # 或访问 https://ollama.ai 下载安装
   ```

2. **下载模型**（选择其一）
   ```bash
   # 推荐：轻量级 1.5B 版本（适合低配置电脑）
   ollama pull qwen2.5:1.5b
   
   # 或使用 7B 版本（需要 8-12GB 显存）
   ollama pull qwen2.5:7b
   ```

3. **启动 Ollama 服务**
   ```bash
   # 方式1：后台运行服务
   ollama serve
   
   # 方式2：直接运行模型（会自动启动服务）
   ollama run qwen2.5:1.5b
   ```

## ▶️ 运行

### 1. 安装依赖
```bash
cd min-rag
pip install -r requirements.txt
```

### 2. 准备文档
将你的文档放在 `data/` 目录下（支持 .docx、.doc、.txt、.md 格式）

### 3. 运行程序

**默认使用 1.5B 版本（推荐）：**
```bash
python app.py
```

**指定使用其他模型：**
```bash
# 使用 1.5B 版本（显式指定）
python app.py --model qwen2.5:1.5b

# 使用 7B 版本（需要 8-12GB 显存）
python app.py --model qwen2.5:7b

# 使用其他模型
python app.py --model deepseek-r1:1.5b
python app.py --model llama3.2
```

**检查系统资源（可选）：**
```bash
python app.py --check-resources
```

## 📊 模型选择建议

| 模型版本 | 参数量 | 显存需求 | 内存需求 | 适用场景 |
|---------|--------|---------|---------|---------|
| **qwen2.5:1.5b** | 15亿 | 2-3 GB | 4-8 GB | ✅ 推荐，适合大多数电脑 |
| **qwen2.5:7b** | 70亿 | 8-12 GB | 16-32 GB | 需要更好性能时使用 |
| **qwen2.5:14b** | 140亿 | 16-24 GB | 32-64 GB | 高性能需求 |

## 💡 使用示例

```bash
# 1. 启动 Ollama（如果还没运行）
ollama serve

# 2. 运行程序（默认使用 1.5B）
python app.py

# 3. 输入问题
请输入你的问题：你的工作经历是什么？

# 4. 查看答案和引用片段
```

## 📝 日志记录

系统会自动记录每次对话的详细信息到日志文件：

- **`logs/rag_conversations.log`**: 对话日志
  - 用户问题
  - 检索到的文档片段
  - 最终的 Prompt
  - AI 的回答

- **`logs/rag_debug.log`**: 调试日志
  - 详细的技术信息
  - 检索过程
  - Prompt 构建过程

日志文件会自动创建在 `logs/` 目录下，方便后续分析和调试。

## 🔧 常见问题 

### Q: 如何确认使用的是 1.5B 版本？
A: 运行程序时会显示 `🤖 使用模型：qwen2.5:1.5b`

### Q: 如何切换模型版本？
A: 使用 `--model` 参数：
```bash
python app.py --model qwen2.5:7b
```

### Q: 提示 "Ollama 服务未运行"？
A: 在另一个终端运行 `ollama serve` 或 `ollama run qwen2.5:1.5b`

### Q: 内存不足怎么办？
A: 使用更小的模型版本（1.5B），或增加系统内存

### Q: 如何查看对话日志？
A: 查看 `logs/rag_conversations.log` 文件，包含所有对话记录

