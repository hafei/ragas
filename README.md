# Ragas + SiliconFlow + Milvus 集成测试

这个项目展示了如何使用 Ragas 评估框架结合 SiliconFlow API 和 Milvus 向量数据库进行 RAG 系统评估。

## 项目结构

```
.
├── README.md                    # 项目说明文档
├── config.json                  # 配置文件
├── test_data.json               # 示例测试数据
├── simple_test.py               # 简化测试脚本
├── ragas_siliconflow_milvus_test.py  # 完整集成测试脚本
├── siliconflow_embeddings.py     # SiliconFlow API 嵌入模型
├── milvus_connector.py          # Milvus 向量数据库连接器
└── json_dataset_extractor.py     # JSON 数据集提取器
```

## 功能特性

- **SiliconFlow API 集成**: 使用 SiliconFlow 的嵌入模型生成文本向量
- **Milvus 向量数据库**: 高效存储和检索向量数据
- **JSON 数据集处理**: 从 JSON 文档自动生成评估数据集
- **Ragas 评估**: 使用多种指标评估 RAG 系统性能
- **端到端测试**: 完整的集成测试流程

## 安装依赖

```bash
# 安装基础依赖
pip install ragas pymilvus requests aiohttp numpy

# 安装 OpenAI 支持（用于评估）
pip install openai

# 可选：安装其他依赖
pip install sentence-transformers  # 用于本地嵌入模型
pip install matplotlib pandas       # 用于结果可视化
```

## 配置设置

### 1. 环境变量设置

```bash
# 设置 SiliconFlow API 密钥
export SILICONFLOW_API_KEY="your-siliconflow-api-key"

# 设置 OpenAI API 密钥（用于评估）
export OPENAI_API_KEY="your-openai-api-key"
```

### 2. 配置文件设置

编辑 `config.json` 文件：

```json
{
  "siliconflow_api_key": "your-siliconflow-api-key",
  "openai_api_key": "your-openai-api-key",
  "json_data_path": "test_data.json",
  "embedding_model": "BAAI/bge-large-zh-v1.5",
  "evaluator_model": "gpt-4o-mini",
  "milvus_host": "localhost",
  "milvus_port": 19530,
  "milvus_collection": "ragas_test_docs",
  "num_samples": 10
}
```

## 使用方法

### 1. 快速测试

运行简化测试脚本验证基本功能：

```bash
python simple_test.py
```

这个脚本会测试：
- SiliconFlow 嵌入功能
- JSON 数据集提取
- 基本集成功能

### 2. 完整集成测试

运行完整的集成测试：

```bash
python ragas_siliconflow_milvus_test.py
```

这个脚本会执行：
- 初始化所有组件
- 设置 Milvus 集合和索引
- 加载文档到向量数据库
- 创建评估数据集
- 运行 Ragas 评估
- 生成评估报告

### 3. 单独使用组件

#### SiliconFlow 嵌入模型

```python
from siliconflow_embeddings import SiliconFlowEmbeddings

# 创建嵌入模型
embeddings = SiliconFlowEmbeddings(api_key="your-api-key")

# 生成嵌入向量
text = "这是一个测试文本"
embedding = embeddings.embed_text(text)

# 批量生成嵌入
texts = ["文本1", "文本2", "文本3"]
embeddings = embeddings.embed_documents(texts)
```

#### Milvus 连接器

```python
from milvus_connector import MilvusConnector
from siliconflow_embeddings import SiliconFlowEmbeddings

# 创建连接器
milvus = MilvusConnector(host="localhost", port=19530)

# 连接到 Milvus
milvus.connect()

# 创建集合
embeddings = SiliconFlowEmbeddings(api_key="your-api-key")
dimension = embeddings.get_embedding_dimension()
milvus.create_collection(dimension)

# 插入文档
documents = [
    {"id": "doc1", "content": "文档内容", "metadata": {"category": "test"}}
]
milvus.insert_documents(documents, embeddings)

# 搜索文档
results = milvus.search("查询文本", embeddings, top_k=5)
```

#### JSON 数据集提取器

```python
from json_dataset_extractor import JSONDatasetExtractor

# 创建提取器
extractor = JSONDatasetExtractor("test_data.json")

# 加载文档
documents = extractor.load_documents()

# 生成查询样本
query_samples = extractor.generate_query_samples(num_samples=10)

# 创建 Ragas 数据集
dataset = extractor.create_ragas_dataset()
```

## 数据格式

### 输入 JSON 格式

```json
[
  {
    "id": "doc1",
    "content": "文档内容",
    "metadata": {
      "category": "concept",
      "difficulty": "easy"
    }
  }
]
```

### 输出数据集格式

生成的数据集包含：
- `question`: 查询问题
- `expected_answer`: 期望答案
- `context`: 上下文文档
- `metadata`: 元数据信息

## 评估指标

Ragas 评估使用以下指标：

- **Context Precision**: 上下文精确度
- **Context Recall**: 上下文召回率
- **Faithfulness**: 忠实度
- **Answer Relevancy**: 答案相关性

## 故障排除

### 常见问题

1. **SiliconFlow API 连接失败**
   - 检查 API 密钥是否正确
   - 确认网络连接正常
   - 验证模型名称是否支持

2. **Milvus 连接失败**
   - 确认 Milvus 服务正在运行
   - 检查主机和端口配置
   - 验证用户名和密码（如果需要）

3. **评估失败**
   - 检查 OpenAI API 密钥
   - 确认有足够的 API 配额
   - 验证模型名称正确

### 调试模式

在脚本中添加调试信息：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展功能

### 添加新的嵌入模型

1. 继承 `BaseRagasEmbedding` 类
2. 实现 `embed_text` 和 `aembed_text` 方法
3. 添加批处理优化

### 添加新的评估指标

1. 继承 Ragas 的基础指标类
2. 实现评估逻辑
3. 集成到评估流程中

### 支持其他向量数据库

1. 创建新的连接器类
2. 实现标准的 CRUD 操作
3. 添加搜索和相似度计算

## 许可证

本项目遵循 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目。

## 联系方式

如有问题，请通过 GitHub Issues 联系。
