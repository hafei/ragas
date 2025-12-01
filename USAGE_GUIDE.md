# Ragas + SiliconFlow + Milvus 使用指南

本指南将帮助您快速上手使用 Ragas 评估框架结合 SiliconFlow API 和 Milvus 向量数据库。

## 🚀 快速开始

### 1. 基础测试（无需 API 密钥）

```bash
python3 basic_test.py
```

这个命令会验证：
- ✅ 文件结构完整性
- ✅ JSON 数据加载
- ✅ 数据格式验证
- ✅ 配置文件正确性

### 2. 完整功能测试（需要 API 密钥）

#### 步骤 1: 设置环境变量

```bash
# SiliconFlow API 密钥（必需）
export SILICONFLOW_API_KEY="your-siliconflow-api-key"

# LLM 配置（使用 SiliconFlow 作为评估 LLM）
export LLM_PROVIDER="siliconflow"
export LLM_BASE_URL="https://api.siliconflow.cn/v1"
export LLM_API_KEY="your-siliconflow-api-key"
export EVALUATOR_MODEL="Qwen/Qwen2.5-7B-Instruct"
```

#### 步骤 2: 安装依赖

```bash
# 创建虚拟环境
python3 -m venv ragas_env
source ragas_env/bin/activate

# 安装基础依赖
pip install requests aiohttp numpy

# 安装完整依赖
pip install ragas pymilvus openai
```

#### 步骤 3: 运行完整测试

```bash
python3 ragas_siliconflow_milvus_test.py
```

## 📁 项目文件说明

### 核心组件

| 文件名 | 功能描述 |
|---------|---------|
| [`siliconflow_embeddings.py`](siliconflow_embeddings.py) | SiliconFlow API 嵌入模型实现 |
| [`milvus_connector.py`](milvus_connector.py) | Milvus 向量数据库连接和操作 |
| [`json_dataset_extractor.py`](json_dataset_extractor.py) | JSON 数据集提取和处理 |

### 测试脚本

| 文件名 | 功能描述 |
|---------|---------|
| [`basic_test.py`](basic_test.py) | 基础功能测试（无需 API 密钥） |
| [`simple_test.py`](simple_test.py) | 简化集成测试 |
| [`ragas_siliconflow_milvus_test.py`](ragas_siliconflow_milvus_test.py) | 完整端到端测试 |

### 配置和数据

| 文件名 | 功能描述 |
|---------|---------|
| [`config.json`](config.json) | 配置文件模板 |
| [`test_data.json`](test_data.json) | 示例测试数据 |

## 🔧 配置说明

### config.json 配置项

```json
{
  "siliconflow_api_key": "your-siliconflow-api-key",    // SiliconFlow API 密钥
  "openai_api_key": "your-openai-api-key",            // OpenAI API 密钥（评估用）
  "json_data_path": "test_data.json",                  // JSON 数据文件路径
  "embedding_model": "BAAI/bge-large-zh-v1.5",       // 嵌入模型名称
  "evaluator_model": "gpt-4o-mini",                  // 评估模型名称
  "milvus_host": "localhost",                         // Milvus 服务器地址
  "milvus_port": 19530,                              // Milvus 服务器端口
  "milvus_user": null,                                // Milvus 用户名（可选）
  "milvus_password": null,                             // Milvus 密码（可选）
  "milvus_collection": "ragas_test_docs",             // Milvus 集合名称
  "num_samples": 10                                    // 生成样本数量
}
```

## 📊 数据格式

### 输入 JSON 格式

```json
[
  {
    "id": "doc1",
    "content": "向量数据库是专门用于存储和查询高维向量数据的数据库。",
    "metadata": {
      "category": "concept",
      "difficulty": "easy"
    }
  }
]
```

### 输出评估结果

```json
{
  "context_precision": 0.8500,
  "context_recall": 0.9200,
  "faithfulness": 0.8800,
  "answer_relevancy": 0.7900
}
```

## 🧪 测试流程

### 基础测试流程

1. **文件结构验证** - 检查所有必需文件是否存在
2. **JSON 数据加载** - 验证数据格式和内容
3. **数据结构分析** - 统计文档信息和类别分布
4. **配置文件验证** - 确认配置项完整性

### 完整测试流程

1. **组件初始化** - 设置嵌入模型、数据库连接等
2. **Milvus 集合设置** - 创建集合和索引
3. **文档加载** - 将文档插入向量数据库
4. **搜索测试** - 验证检索功能
5. **数据集生成** - 创建评估数据集
6. **Ragas 评估** - 运行多指标评估

## 🎯 使用场景

### 场景 1: 评估不同嵌入模型

```python
from siliconflow_embeddings import SiliconFlowEmbeddings

# 测试不同模型
models = [
    "BAAI/bge-large-zh-v1.5",
    "BAAI/bge-small-zh-v1.5",
    "shibing624/text2vec-large-chinese"
]

for model in models:
    embeddings = SiliconFlowEmbeddings(api_key=api_key, model_name=model)
    # 运行评估...
```

### 场景 2: 比较检索策略

```python
from milvus_connector import MilvusConnector

# 测试不同索引类型
index_types = ["HNSW", "IVF_FLAT", "IVF_PQ"]

for index_type in index_types:
    milvus.create_index(index_type=index_type)
    # 运行评估...
```

### 场景 3: 批量评估

```python
from json_dataset_extractor import JSONDatasetExtractor

# 生成不同规模的测试集
sample_sizes = [10, 50, 100]

for size in sample_sizes:
    extractor.generate_query_samples(num_samples=size)
    # 运行评估...
```

## 🐛 故障排除

### 常见问题及解决方案

#### 1. SiliconFlow API 连接失败

**错误**: `SiliconFlow API 请求失败`

**解决方案**:
- 检查 API 密钥是否正确
- 确认网络连接正常
- 验证模型名称是否支持

#### 2. Milvus 连接失败

**错误**: `连接 Milvus 失败`

**解决方案**:
- 确认 Milvus 服务正在运行
- 检查主机和端口配置
- 验证防火墙设置

#### 3. 评估失败

**错误**: `Ragas 评估失败`

**解决方案**:
- 检查 OpenAI API 密钥
- 确认有足够的 API 配额
- 验证模型名称正确

#### 4. 依赖安装失败

**错误**: `ModuleNotFoundError`

**解决方案**:
```bash
# 使用虚拟环境
python3 -m venv ragas_env
source ragas_env/bin/activate

# 升级 pip
pip install --upgrade pip

# 重新安装依赖
pip install ragas pymilvus openai requests aiohttp numpy
```

## 📈 性能优化建议

### 1. 嵌入优化

- 使用批量处理减少 API 调用
- 缓存常用文本的嵌入结果
- 选择合适的模型大小

### 2. 数据库优化

- 合理设置索引参数
- 定期优化集合
- 监控内存使用情况

### 3. 评估优化

- 使用较小的评估模型（如 gpt-4o-mini）
- 采样评估而非全量评估
- 并行处理评估任务

## 🔗 扩展集成

### 添加新的嵌入模型

```python
from siliconflow_embeddings import SiliconFlowEmbeddings

class CustomEmbeddings(SiliconFlowEmbeddings):
    def __init__(self, api_key, model_name="custom-model"):
        super().__init__(api_key, model_name)
        # 自定义初始化逻辑
```

### 添加新的评估指标

```python
from ragas.metrics.base import Metric

class CustomMetric(Metric):
    def score(self, dataset):
        # 自定义评估逻辑
        return scores
```

## 📚 参考资料

- [Ragas 官方文档](https://docs.ragas.io/)
- [SiliconFlow API 文档](https://docs.siliconflow.cn/)
- [Milvus 官方文档](https://milvus.io/docs/)
- [向量数据库最佳实践](https://www.pinecone.io/learn/vector-database-best-practices)

## 🤝 贡献指南

欢迎贡献代码和改进建议！

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 📞 获取帮助

如果遇到问题：

1. 查看本文档的故障排除部分
2. 检查 GitHub Issues
3. 创建新的 Issue 描述问题

---

**祝您使用愉快！** 🎉