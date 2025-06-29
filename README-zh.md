# ChrysalisMCP: 一个检索增强生成 (RAG) 平台

ChrysalisMCP (模型社区平台) 是一个强大的命令行界面 (CLI) 应用程序，旨在通过先进的检索增强生成 (RAG) 功能和持久化记忆系统来增强大型语言模型 (LLM)。它允许 LLM 利用本地知识库提供更准确、更具上下文感知的回答，模拟用于长上下文处理的“全局记忆”。

## 功能

-   **混合检索**: 结合语义搜索 (FAISS) 和关键词搜索 (BM25)，实现全面的文档检索。
-   **多模态 RAG**: 从各种文档类型中提取文本，包括：
    -   纯文本 (`.txt`)
    -   PDF (`.pdf`)
    -   Microsoft Word (`.docx`)
    -   Microsoft PowerPoint (`.pptx`)
    -   图像 (`.png`, `.jpg`, `.jpeg`) 通过 OCR (需要 Tesseract OCR 引擎)。
-   **LLM 驱动的线索生成**: 根据原始查询和全局记忆，使用 LLM 生成更详细的“线索”，从而增强检索查询。
-   **全局记忆**: 维护整个知识库的持久化、摘要性概览，在索引构建期间自动更新。
-   **持久化用户记忆**: 将特定的用户相关事实保存到长期记忆中。
-   **可扩展的提示系统**: 轻松定义和扩展各种 LLM 提示（例如，代码生成、摘要、解释）。
-   **工具集成**: 无缝集成 Google 搜索、网页抓取和 shell 命令等外部工具。
-   **模块化设计**: 采用清晰的职责分离设计，便于维护和扩展。

## 设置

按照以下步骤在本地设置和运行 ChrysalisMCP。

### 1. 克隆仓库

```bash
git clone <您的 GitHub 仓库 URL>
cd MCP
```

### 2. 安装依赖

确保您已安装 Python (3.9+) 和 pip。然后，安装所需的 Python 包：

```bash
pip install -r requirements.txt
```

### 3. 安装 Tesseract OCR (用于图像支持)

如果您计划在知识库中使用图像文件 (`.png`, `.jpg`, `.jpeg`)，则**必须**在您的系统上安装 Tesseract OCR 引擎。请按照您操作系统的安装说明进行操作：

-   **Windows**: [Windows 版 Tesseract-OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html#windows)
-   **macOS**: `brew install tesseract`
-   **Linux**: `sudo apt-get install tesseract-ocr`

### 4. 配置 API 密钥

ChrysalisMCP 使用 Google 的生成式 AI 模型进行全局记忆摘要，并可能用于线索生成。请将您的 Google API 密钥设置为环境变量：

```bash
# 适用于 Linux/macOS
export GOOGLE_API_KEY="您的 Google API 密钥"

# 适用于 Windows (命令提示符)
set GOOGLE_API_KEY="您的 Google API 密钥"

# 适用于 Windows (PowerShell)
$env:GOOGLE_API_KEY="您的 Google API 密钥"
```

如果您计划使用 Google 搜索工具，您还需要在 `default_api.py` 中配置 `GOOGLE_SEARCH_API_KEY` 和 `GOOGLE_SEARCH_CX`。

### 5. 准备知识库

将您的 `.txt`、`.pdf`、`.docx`、`.pptx`、`.png`、`.jpg` 和 `.jpeg` 文件放入 `mcp_resources/` 目录。这些文档将被索引并由 RAG 系统使用。

### 6. 构建 FAISS 索引和全局记忆

运行 `build_index.py` 脚本以处理您的文档，创建 FAISS 向量索引，并生成全局记忆摘要。此步骤需要一些时间，特别是对于大型知识库或首次下载嵌入模型时。

```bash
python build_index.py
```

## 使用方法

### 1. 启动 MCP 服务器

```bash
python mcp_server.py
```

服务器默认将在 `http://0.0.0.0:8000` 启动。

### 2. 与 CLI 交互 (示例: 使用 `ask-rag`)

服务器运行后，您可以通过 CLI 与其交互。例如，使用 `ask-rag` 提示：

```bash
# 首次调用 ask-rag (将提示生成线索)
# CLI 客户端将处理交互以从 LLM 获取线索
# 然后使用生成的线索重新调用 ask-rag。
# 示例 (概念性，实际 CLI 交互可能有所不同):
# user: ask-rag --query "关于 RAG 的文档主要内容是什么？"
# CLI: (内部调用 LLM 生成线索)
# CLI: (使用生成的线索重新调用 ask-rag)
# CLI: (显示答案)
```

**注意**: `mcp_server.py` 中当前的 `ask-rag` 实现是为能够处理多步交互（即，接收线索提示，将其发送到 LLM，然后使用生成的线索重新调用 `ask-rag`）的客户端设计的。简单的直接 CLI 调用可能无法在没有更复杂的客户端的情况下完全演示此流程。

## 项目结构

```
MCP/
├── default_api.py          # 核心工具实现（文件系统、网络、记忆等）
├── mcp_calculator.py       # 计算器工具实现
├── mcp_server.py           # 主服务器应用程序、提示定义和工具封装
├── build_index.py          # 构建 FAISS 索引和全局记忆的脚本
├── requirements.txt        # Python 依赖
├── .gitignore              # Git 忽略文件
├── mcp_resources/          # 知识库文档目录
│   ├── your_document.txt
│   ├── another_doc.pdf
│   └── image_with_text.png
├── faiss_index/            # 生成的 FAISS 索引文件 (Git 忽略)
└── chunks.json             # 文档块的元数据 (Git 忽略)
```

## 贡献

欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 获取指南。

## 许可证

本项目采用 [MIT 许可证](LICENSE) 授权。