import os
import sys
import requests
import warnings
import logging
from pathlib import Path

# 禁用 ChromaDB 遥测功能（避免错误信息）
# 必须在导入 Chroma 之前设置
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# 忽略 ChromaDB 遥测相关的警告
warnings.filterwarnings(
    "ignore",
    message=".*telemetry.*",
    category=UserWarning
)

# 以下导入必须在环境变量设置之后
# 以下导入必须在环境变量设置之后
from langchain_community.document_loaders import (  # noqa: E402
    Docx2txtLoader
)
from langchain_text_splitters import (  # noqa: E402
    RecursiveCharacterTextSplitter
)
from langchain_community.embeddings import (  # noqa: E402
    SentenceTransformerEmbeddings
)
from langchain_community.vectorstores import Chroma  # noqa: E402
from langchain_community.llms import Ollama  # noqa: E402
from langchain.prompts import PromptTemplate  # noqa: E402
from langchain.chains import RetrievalQA  # noqa: E402


def setup_logging(log_dir="logs"):
    """
    配置日志系统
    
    日志会记录到：
    - logs/rag_conversations.log: 对话日志（问题、答案、检索内容等）
    - logs/rag_debug.log: 调试日志（详细的 prompt、向量信息等）
    
    Returns:
        logger: 配置好的日志记录器
    """
    # 创建日志目录
    Path(log_dir).mkdir(exist_ok=True)
    
    # 配置主日志（对话记录）
    conversation_logger = logging.getLogger("conversation")
    conversation_logger.setLevel(logging.INFO)
    
    # 如果已经配置过，避免重复添加 handler
    if not conversation_logger.handlers:
        conversation_handler = logging.FileHandler(
            os.path.join(log_dir, "rag_conversations.log"),
            encoding="utf-8"
        )
        conversation_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        conversation_logger.addHandler(conversation_handler)
    
    # 配置调试日志（详细技术信息）
    debug_logger = logging.getLogger("debug")
    debug_logger.setLevel(logging.DEBUG)
    
    if not debug_logger.handlers:
        debug_handler = logging.FileHandler(
            os.path.join(log_dir, "rag_debug.log"),
            encoding="utf-8"
        )
        debug_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - [DEBUG] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        debug_logger.addHandler(debug_handler)
    
    return conversation_logger, debug_logger


"""
================================================================================
RAG (Retrieval-Augmented Generation) 检索增强生成系统工作流程
================================================================================

RAG 系统通过以下步骤实现基于文档的智能问答：

【阶段1：文档预处理（离线，首次运行）】
1. 文档加载 (Document Loading)
   - 从文件系统加载文档（Word、PDF、TXT等）
   - 将文档转换为统一的文本格式

2. 文本分块 (Text Chunking)
   - 将长文档切分成多个小片段（chunks）
   - 每个片段包含部分重叠，保持上下文连贯性
   - 目的：便于检索和匹配

3. 向量化 (Embedding)
   - 使用嵌入模型（如 SentenceTransformer）将文本转换为向量
   - 每个文本块变成一个高维向量（如 384 维）
   - 语义相似的文本会产生相似的向量

4. 向量存储 (Vector Storage)
   - 将所有文档块的向量存储到向量数据库（ChromaDB）
   - 建立索引，支持快速相似度搜索
   - 持久化存储，避免重复处理

【阶段2：问答流程（在线，每次查询）】
5. 问题向量化
   - 将用户问题转换为向量（使用相同的嵌入模型）

6. 相似度检索 (Retrieval)
   - 在向量数据库中搜索与问题最相似的文档片段
   - 返回 Top-K 个最相关的文档块（如 Top-4）

7. 上下文增强 (Augmentation)
   - 将检索到的文档片段作为上下文
   - 与用户问题一起构建提示词（Prompt）

8. 生成答案 (Generation)
   - 将增强后的提示词输入大语言模型（LLM）
   - LLM 基于检索到的上下文生成答案
   - 返回答案和引用来源

================================================================================
优势：
- 可以回答文档中的具体内容（LLM 本身不知道文档内容）
- 答案有据可查（可以追溯到源文档）
- 支持长文档（通过分块处理）
- 可以更新知识（只需更新向量库，无需重新训练模型）
================================================================================
"""


def load_word_document(file_path):
    """
    加载 Word 文档
    - .docx 格式：使用 Docx2txtLoader（不需要 LibreOffice）
    - .doc 格式：需要 LibreOffice，如果未安装会给出提示
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.docx':
        # 使用 Docx2txtLoader 加载 .docx 文件（不需要 LibreOffice）
        loader = Docx2txtLoader(file_path)
        return loader.load()
    elif file_ext == '.doc':
        # .doc 格式需要 LibreOffice
        try:
            from langchain_community.document_loaders import (
                UnstructuredWordDocumentLoader
            )
            loader = UnstructuredWordDocumentLoader(file_path)
            return loader.load()
        except FileNotFoundError as e:
            if 'soffice' in str(e):
                raise ValueError(
                    "❌ 检测到 .doc 格式文件，需要安装 LibreOffice。\n\n"
                    "📋 解决方案（选择其一）：\n"
                    "1. 【推荐】将文件转换为 .docx 格式\n"
                    "   - 用 Word 或 LibreOffice 打开文件\n"
                    "   - 另存为 .docx 格式\n"
                    "   - 更新代码中的文件路径\n\n"
                    "2. 安装 LibreOffice（macOS）：\n"
                    "   brew install --cask libreoffice\n\n"
                    f"当前文件：{file_path}"
                ) from e
            raise
    else:
        raise ValueError(f"不支持的文件格式：{file_ext}")


def build_vectorstore():
    """
    【RAG 阶段1：构建向量库】
    
    这是 RAG 系统的离线预处理阶段，只需要在首次运行或文档更新时执行。
    构建好的向量库会持久化保存，后续可以直接使用。
    
    流程：
    1. 文档加载 → 2. 文本分块 → 3. 向量化 → 4. 向量存储
    """
    # ========== 步骤1：文档加载 ==========
    # 从文件系统加载文档，转换为 Document 对象列表
    # Document 对象包含 page_content（文本内容）和 metadata（元数据）
    file_path = "data/简历_互联网相关业务版-08.docx"
    docs = load_word_document(file_path)
    print(f"✅ 已加载文档，共 {len(docs)} 页")

    # ========== 步骤2：文本分块 ==========
    # 将长文档切分成多个小片段，便于后续检索
    # chunk_size=500: 每个片段约 500 个字符
    # chunk_overlap=50: 相邻片段重叠 50 个字符，保持上下文连贯性
    # 例如：文档 "ABCDEFGHIJKLMN" 可能被切分为：
    #   Chunk1: "ABCDEFGHIJ" (0-500)
    #   Chunk2: "FGHIJKLMNO" (450-950, 与 Chunk1 重叠 50)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ 文档已分块，共 {len(chunks)} 个片段")

    # ========== 步骤3：向量化 ==========
    # 使用嵌入模型将文本转换为向量
    # all-MiniLM-L6-v2: 轻量级模型，输出 384 维向量
    # 语义相似的文本会产生相似的向量（余弦相似度高）
    # 例如："我喜欢苹果" 和 "我爱吃苹果" 的向量会很接近
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    print("✅ 嵌入模型已加载")

    # ========== 步骤4：向量存储 ==========
    # 将所有文档块的向量存储到 ChromaDB 向量数据库
    # persist_directory: 指定持久化目录，向量库会保存到磁盘
    # 这样下次运行就不需要重新处理文档了
    vectordb = Chroma.from_documents(
        chunks, embeddings, persist_directory="vectordb"
    )
    # 持久化保存向量库
    vectordb.persist()
    print("✅ 向量库已构建并保存")
    return vectordb


def check_system_resources():
    """检查系统资源并给出模型选择建议"""
    try:
        import psutil
        # 获取内存信息（GB）
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        available_memory_gb = memory.available / (1024**3)

        # 尝试获取 GPU 信息（如果有）
        gpu_info = None
        try:
            import subprocess
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader"
                ],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                gpu_memory_mb = int(result.stdout.strip().split()[0])
                gpu_info = f"{gpu_memory_mb / 1024:.1f} GB"
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

        print("\n💻 系统资源信息：")
        print(f"   总内存：{total_memory_gb:.1f} GB")
        print(f"   可用内存：{available_memory_gb:.1f} GB")
        if gpu_info:
            print(f"   GPU 显存：{gpu_info}")
        else:
            print("   GPU：未检测到 NVIDIA GPU（将使用 CPU）")

        print("\n📊 模型选择建议：")
        if total_memory_gb >= 32 and gpu_info:
            print("   ✅ 推荐使用：qwen2.5:7b（性能更好）")
        elif total_memory_gb >= 16:
            print("   ✅ 推荐使用：qwen2.5:1.5b（当前配置）")
            print("   ⚠️  如果显存充足，可以尝试：qwen2.5:7b")
        else:
            print("   ✅ 推荐使用：qwen2.5:1.5b（轻量级，适合低配置）")
        print()

    except ImportError:
        print("\n💡 提示：安装 psutil 可以查看详细的系统资源信息")
        print("   运行：pip install psutil\n")
    except Exception as e:
        print(f"\n⚠️  无法检查系统资源：{str(e)}\n")


def check_ollama_connection(base_url="http://localhost:11434"):
    """检查 Ollama 服务是否运行"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            return True, None
        return False, f"Ollama 服务返回错误状态码: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "无法连接到 Ollama 服务"
    except Exception as e:
        return False, f"检查 Ollama 连接时出错: {str(e)}"


def load_qa_chain(model_name="qwen2.5:1.5b"):
    """
    【RAG 阶段2：加载问答链】
    
    构建 RAG 系统的核心组件，将检索器和生成器组合在一起。
    每次用户提问时，这个链会自动执行：检索 → 增强 → 生成
    
    Args:
        model_name: Ollama 模型名称，默认为 qwen2.5:1.5b
                   可选值：
                   - qwen2.5:1.5b (轻量级，推荐)
                   - qwen2.5:7b (需要 8-12GB 显存)
                   - deepseek-r1:1.5b
                   - llama3.2 等
    """
    # ========== 加载向量数据库 ==========
    # 从磁盘加载之前构建好的向量库
    # 使用相同的嵌入模型，确保向量空间一致
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectordb = Chroma(
        persist_directory="vectordb",
        embedding_function=embedding_function
    )
    
    # ========== 创建检索器 ==========
    # 检索器负责根据问题找到最相关的文档片段
    # 默认使用相似度搜索，返回 Top-K 个最相似的文档块
    retriever = vectordb.as_retriever()
    # 可以设置检索参数，例如：
    # retriever = vectordb.as_retriever(search_kwargs={"k": 4})  # 返回 Top-4

    # ========== 初始化大语言模型 ==========
    # LLM 负责基于检索到的上下文生成答案
    # 使用本地部署的模型（通过 Ollama），保护数据隐私
    llm = Ollama(model=model_name)

    # ========== 构建提示词模板 ==========
    # 定义如何将检索到的文档和用户问题组合成提示词
    # {context}: 检索到的文档片段（由检索器自动填充）
    # {question}: 用户的问题（由用户输入）
    prompt_template = PromptTemplate(
        template=(
            "参考以下检索到的内容回答问题。\n\n"
            "检索内容：\n{context}\n\n"
            "问题：{question}\n\n"
            "请给出简洁答案。"
        ),
        input_variables=["context", "question"]
    )

    # ========== 组装检索增强生成链 ==========
    # RetrievalQA 链将检索和生成过程串联起来：
    # 1. 接收用户问题
    # 2. 使用检索器找到相关文档片段
    # 3. 将文档片段和问题组合成提示词
    # 4. 调用 LLM 生成答案
    # 5. 返回答案和源文档引用
    chain = RetrievalQA.from_chain_type(
        llm=llm,                    # 生成模型
        retriever=retriever,        # 检索器
        return_source_documents=True,  # 返回源文档，便于追溯答案来源
        chain_type_kwargs={"prompt": prompt_template}  # 自定义提示词模板
    )
    # 返回 chain 和 prompt_template，便于日志记录
    return chain, prompt_template, retriever


if __name__ == "__main__":
    # 解析命令行参数
    model_name = "qwen2.5:1.5b"  # 默认使用 1.5B 版本
    
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model_name = sys.argv[idx + 1]
        else:
            print("❌ --model 参数需要指定模型名称")
            print("   例如：python app.py --model qwen2.5:1.5b")
            exit(1)
    
    if "--check-resources" in sys.argv:
        check_system_resources()

    # 检查 Ollama 服务是否运行
    is_connected, error_msg = check_ollama_connection()
    if not is_connected:
        print("❌ Ollama 服务未运行！")
        print(f"错误信息：{error_msg}\n")
        print("📋 解决方案：")
        print("1. 启动 Ollama 服务：")
        print("   - 在终端运行：ollama serve")
        print("   - 或者直接运行：ollama run qwen2.5:1.5b")
        print("\n2. 确保已安装 Ollama：")
        print("   - 访问 https://ollama.ai 下载安装")
        print("   - macOS: brew install ollama")
        exit(1)

    # ========== 初始化日志系统 ==========
    conv_logger, debug_logger = setup_logging()
    conv_logger.info("="*80)
    conv_logger.info(f"RAG 系统启动 - 模型: {model_name}")
    debug_logger.debug(f"系统启动 - 模型: {model_name}")

    # ========== RAG 系统初始化 ==========
    # 检查向量库是否存在，如果不存在则构建（首次运行）
    if not os.path.exists("vectordb"):
        print("🔧 第一次运行，开始构建向量库...")
        print("   这可能需要几分钟时间，请耐心等待...")
        build_vectorstore()
        print("\n✅ 向量库构建完成！")

    # 加载问答链（包含检索器和生成器）
    print(f"🤖 使用模型：{model_name}")
    chain, prompt_template, retriever = load_qa_chain(model_name=model_name)

    print("🟢 Mini-RAG 文档问答系统已启动（输入 exit 退出）")
    print("\n" + "="*60)
    print("💡 RAG 工作流程：")
    print("   1. 您输入问题")
    print("   2. 系统在文档中检索相关内容")
    print("   3. 将相关内容与问题一起发送给 AI")
    print("   4. AI 基于文档内容生成答案")
    print("="*60 + "\n")
    
    # ========== 交互式问答循环 ==========
    while True:
        query = input("\n请输入你的问题：")
        if query.lower() == "exit":
            print("\n👋 再见！")
            break

        try:
            # ========== 记录用户问题 ==========
            conv_logger.info(f"\n{'='*80}")
            conv_logger.info(f"【用户问题】{query}")
            debug_logger.debug(f"用户问题: {query}")
            
            # ========== RAG 核心流程：检索 + 生成 ==========
            # chain.invoke() 会自动执行以下步骤：
            # 
            # 【步骤1：问题向量化】
            #   将用户问题转换为向量（使用嵌入模型）
            #
            # 【步骤2：相似度检索】
            #   在向量数据库中搜索与问题向量最相似的文档片段
            #   默认返回 Top-K 个最相关的文档块
            #
            # 【步骤3：上下文增强】
            #   将检索到的文档片段作为上下文，与问题组合：
            #   "参考以下检索到的内容回答问题。
            #    检索内容：[文档片段1] [文档片段2] ...
            #    问题：[用户问题]
            #    请给出简洁答案。"
            #
            # 【步骤4：生成答案】
            #   将增强后的提示词发送给 LLM
            #   LLM 基于检索到的上下文生成答案
            #
            # 【步骤5：返回结果】
            #   result["result"]: 生成的答案
            #   result["source_documents"]: 检索到的源文档片段
            
            # ========== 步骤2：手动执行检索（用于日志记录）==========
            # 先检索文档片段，记录到日志
            retrieved_docs = retriever.get_relevant_documents(query)
            conv_logger.info(f"\n【检索到的文档片段数量】{len(retrieved_docs)}")
            debug_logger.debug(f"检索到 {len(retrieved_docs)} 个文档片段")
            
            # 记录检索到的文档内容
            for i, doc in enumerate(retrieved_docs, 1):
                doc_content = doc.page_content[:500]  # 前500字符
                conv_logger.info(f"\n【检索片段 {i}/{len(retrieved_docs)}】")
                conv_logger.info(f"{doc_content}...")
                debug_logger.debug(f"检索片段 {i}: {doc_content[:200]}...")
            
            # ========== 步骤3：构建最终的 Prompt ==========
            # 将检索到的文档组合成上下文
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # 使用 prompt_template 格式化最终的 prompt
            final_prompt = prompt_template.format(
                context=context,
                question=query
            )
            
            # ========== 记录最终的 Prompt ==========
            conv_logger.info("\n【最终 Prompt】")
            conv_logger.info(final_prompt)
            debug_logger.debug(f"最终 Prompt:\n{final_prompt}")
            
            # ========== 步骤4：调用链生成答案 ==========
            result = chain.invoke({"query": query})
            
            # ========== 记录最终响应 ==========
            answer = result["result"]
            conv_logger.info("\n【AI 回答】")
            conv_logger.info(answer)
            conv_logger.info(f"\n{'='*80}\n")
            debug_logger.debug(f"AI 回答: {answer}")
            
            # 显示答案
            print("\n📘 答案：\n", answer)

            # 可选：显示检索到的源文档片段（已注释）
            # 取消注释可以查看答案的来源，便于验证答案的准确性
            # print("\n📚 引用片段：")
            # for doc in result["source_documents"]:
            #     print("-", doc.page_content[:200], "...")
        except requests.exceptions.ConnectionError:
            print("\n❌ 连接 Ollama 失败，请确保 Ollama 服务正在运行")
            print("   运行命令：ollama serve")
            break
        except Exception as e:
            print(f"\n❌ 发生错误：{str(e)}")
            break
