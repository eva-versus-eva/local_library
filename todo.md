# 📝 TODO List 

## 1. Integrate Multi-Agent Application (Optional)
- Attempt to connect with existing multi-agent frameworks or APIs to enhance Q&A and collaboration capabilities.

## 2. Support Local Embedding Models
- Allow users to select or switch to fully offline local embedding models (e.g., ONNX, llama.cpp, etc.).

## 3. Upload Enhancements
- **Progress bar**: Display processing progress (chunking, vectorization) when uploading large files.
- **Concurrent uploads**: Support uploading multiple files simultaneously with asynchronous processing.
- **Extended formats**: Add support for common document formats such as `.docx`, `.html`, `.epub`, etc.

## 4. Chat History Persistence
- Save conversation history to a local file (e.g., JSON) and automatically restore it after app restart (can also be achieved via integration with a multi-agent app).

## 5. Boundary Testing & Warnings
- Add unit tests and boundary condition tests (empty files, oversized files, incorrect encoding, etc.).
- Provide user-friendly warnings for potential issues (e.g., API rate limiting, vector store write failures).

## 6. Encapsulate Vector Store Operations into a Class
- Encapsulate Chroma's add, delete, query, and metadata management into a `VectorStoreManager` class to improve code maintainability.

## 7. Support Multiple Knowledge Bases (Library Abstraction)
- Allow users to create multiple subfolders under `./library` (e.g., `./library/personal_docs`, `./library/project_materials`), each as an independent knowledge base that can be switched between.

## 8. Delete Files or Entire Knowledge Bases
- Provide UI buttons to support:
  - Deleting a single processed file (remove from the vector store).
  - Deleting an entire knowledge base (clear the corresponding Chroma collection and metadata records).

## 9. Improve Logging System
- Use the Python `logging` module to record key operations (file processing, retrieval latency, API call status) for easier debugging and performance analysis.


# 📝 TODO List

## 1. 集成多智能体应用（可选）
- 尝试连接现有的多智能体框架或 API，增强问答与协作能力。

## 2. 支持本地 Embedding 模型
- 允许用户选择或切换至完全离线的本地 Embedding 模型（如 ONNX、llama.cpp 等）。

## 3. 上传功能增强
- **进度条**：大文件上传时显示处理进度（分块、向量化）。
- **多文件并发**：支持同时上传多个文件并异步处理。
- **扩展格式**：增加对 `.docx`、`.html`、`.epub` 等常见文档格式的支持。

## 4. 聊天记录持久化
- 将对话历史保存至本地文件（如 JSON），应用重启后自动恢复（也可通过与多智能体应用集成实现）。

## 5. 边界测试与警告
- 增加单元测试和边界条件测试（空文件、超大文件、错误编码等）。
- 对潜在问题（如 API 限流、向量库写入失败）给出用户友好的警告提示。

## 6. 向量库操作封装为类
- 将 Chroma 的增、删、查以及元数据管理封装成一个 `VectorStoreManager` 类，提高代码可维护性。

## 7. 支持多知识库（Library 抽象）
- 允许用户在 `./library` 下创建多个子文件夹（如 `./library/个人文档`、`./library/项目资料`），每个子文件夹作为一个独立知识库，可切换使用。

## 8. 删除文件或整个知识库
- 提供 UI 按钮，支持：
  - 删除单个已处理文件（从向量库中移除）。
  - 删除整个知识库（清空对应的 Chroma 集合及元数据记录）。

## 9. 完善日志系统
- 使用 Python `logging` 模块记录关键操作（文件处理、检索耗时、API 调用状态），便于调试和性能分析。