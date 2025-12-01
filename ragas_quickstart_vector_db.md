# Ragas 向量数据库质量测试快速入门

## 5分钟快速开始

### 1. 安装与设置

```bash
# 安装 Ragas
pip install ragas

# 设置 API 密钥
export OPENAI_API_KEY="your-openai-key"
```

### 2. 准备测试数据

```python
from ragas import Dataset

# 创建测试数据集
dataset = Dataset(name="vector_db_test", backend="local/csv", root_dir="./data")

# 添加测试样本
test_samples = [
    {
        "question": "什么是向量数据库？",
        "expected_answer": "向量数据库是专门用于存储和查询高维向量数据的数据库。",
        "metadata": {"category": "concept", "difficulty": "easy"}
    },
    {
        "question": "向量数据库如何提高检索效率？",
        "expected_answer": "向量数据库通过近似最近邻搜索算法如 HNSW、IVF 等提高检索效率。",
        "metadata": {"category": "technical", "difficulty": "medium"}
    }
]

for sample in test_samples:
    dataset.append(sample)

dataset.save()
```

### 3. 运行基础评估

```python
from ragas import evaluate
from ragas.metrics import ContextPrecision, ContextRecall
from ragas.llms import llm_factory
from openai import AsyncOpenAI

# 设置评估器
client = AsyncOpenAI()
evaluator_llm = llm_factory("gpt-4o-mini", client=client)

# 定义评估指标
metrics = [
    ContextPrecision(llm=evaluator_llm),
    ContextRecall(llm=evaluator_llm)
]

# 运行评估
result = evaluate(dataset=dataset, metrics=metrics)
print(f"上下文精确度: {result['context_precision']}")
print(f"上下文召回率: {result['context_recall']}")
```

## 评估向量数据库性能

### 1. 比较不同嵌入模型

```python
from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
import openai

# 设置测试生成器
generator_llm = ChatOpenAI(model="gpt-4o-mini")
critic_llm = ChatOpenAI(model="gpt-4o")
openai_client = openai.OpenAI()
embeddings = OpenAIEmbeddings(client=openai_client)

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# 从文档生成测试数据
testset = generator.generate_with_langchain_docs(your_documents, testset_size=20)

# 评估不同嵌入模型
def evaluate_embedding_model(embedding_model, model_name):
    # 构建使用该嵌入的检索器
    retriever = build_retriever_with_embeddings(your_documents, embedding_model)
    
    # 评估检索性能
    result = evaluate(
        dataset=testset.to_evaluation_dataset(),
        metrics=[ContextPrecision(llm=evaluator_llm), ContextRecall(llm=evaluator_llm)]
    )
    
    print(f"{model_name} - 精确度: {result['context_precision']:.3f}, 召回率: {result['context_recall']:.3f}")
    return result

# 比较 OpenAI 和 BGE 嵌入
from langchain.embeddings import HuggingFaceEmbeddings

openai_result = evaluate_embedding_model(OpenAIEmbeddings(client=openai_client), "OpenAI")
bge_result = evaluate_embedding_model(HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"), "BGE")
```

### 2. 测试不同分块策略

```python
def test_chunking_strategies(documents, testset):
    """测试不同分块策略对检索效果的影响"""
    
    strategies = {
        "fixed_256": {"chunk_size": 256, "chunk_overlap": 0},
        "fixed_512": {"chunk_size": 512, "chunk_overlap": 50},
        "fixed_1024": {"chunk_size": 1024, "chunk_overlap": 100}
    }
    
    results = {}
    
    for strategy_name, params in strategies.items():
        # 使用特定分块策略构建索引
        retriever = build_retriever_with_chunking(documents, **params)
        
        # 评估性能
        result = evaluate(
            dataset=testset.to_evaluation_dataset(),
            metrics=[ContextPrecision(llm=evaluator_llm), ContextRecall(llm=evaluator_llm)]
        )
        
        results[strategy_name] = {
            "context_precision": result['context_precision'],
            "context_recall": result['context_recall']
        }
        
        print(f"{strategy_name}: 精确度={result['context_precision']:.3f}, 召回率={result['context_recall']:.3f}")
    
    return results

# 运行分块测试
chunking_results = test_chunking_strategies(your_documents, testset)
```

### 3. 评估检索器类型

```python
def compare_retriever_types(documents, testset):
    """比较不同检索器类型的性能"""
    
    # BM25 检索器
    bm25_retriever = build_bm25_retriever(documents)
    bm25_result = evaluate_retriever(bm25_retriever, testset, "BM25")
    
    # 向量检索器
    vector_retriever = build_vector_retriever(documents, OpenAIEmbeddings(client=openai_client))
    vector_result = evaluate_retriever(vector_retriever, testset, "Vector")
    
    # 混合检索器
    hybrid_retriever = build_hybrid_retriever(documents, OpenAIEmbeddings(client=openai_client))
    hybrid_result = evaluate_retriever(hybrid_retriever, testset, "Hybrid")
    
    return {
        "BM25": bm25_result,
        "Vector": vector_result,
        "Hybrid": hybrid_result
    }

def evaluate_retriever(retriever, testset, retriever_name):
    """评估单个检索器"""
    result = evaluate(
        dataset=testset.to_evaluation_dataset(),
        metrics=[ContextPrecision(llm=evaluator_llm), ContextRecall(llm=evaluator_llm)]
    )
    
    print(f"{retriever_name} - 精确度: {result['context_precision']:.3f}, 召回率: {result['context_recall']:.3f}")
    return result
```

## 实验管理

### 1. 创建可重复实验

```python
from ragas import experiment
import asyncio

@experiment()
async def evaluate_vector_db_config(row, retriever_config, llm):
    """评估不同向量数据库配置的实验函数"""
    question = row["question"]
    
    # 使用特定配置构建检索器
    retriever = build_retriever_with_config(retriever_config)
    
    # 查询检索器
    retrieved_docs = await retriever.aretrieve(question, top_k=5)
    
    # 评估检索质量
    context_precision_score = await ContextPrecision(llm=llm).ascore(
        user_input=question,
        reference=row["expected_answer"],
        retrieved_contexts=[doc.page_content for doc in retrieved_docs]
    )
    
    context_recall_score = await ContextRecall(llm=llm).ascore(
        user_input=question,
        retrieved_contexts=[doc.page_content for doc in retrieved_docs],
        reference=row["expected_answer"]
    )
    
    return {
        **row,
        "retrieved_docs": [doc.page_content for doc in retrieved_docs],
        "context_precision": context_precision_score.value,
        "context_recall": context_recall_score.value,
        "config": retriever_config,
        "experiment_name": f"vector_db_test_{retriever_config['name']}"
    }

# 运行实验比较不同配置
configs = [
    {"name": "openai_1536", "embedding_model": "text-embedding-3-large", "dim": 1536},
    {"name": "bge_1024", "embedding_model": "BAAI/bge-large-en-v1.5", "dim": 1024},
    {"name": "e5_1024", "embedding_model": "intfloat/e5-large-v2", "dim": 1024}
]

for config in configs:
    results = await evaluate_vector_db_config.arun(
        dataset, 
        retriever_config=config,
        llm=evaluator_llm
    )
    print(f"完成 {config['name']} 配置评估")
```

### 2. 结果分析与可视化

```python
import pandas as pd
import matplotlib.pyplot as plt

def analyze_experiment_results(experiment_files):
    """分析实验结果并生成可视化"""
    
    all_results = []
    
    for file in experiment_files:
        df = pd.read_csv(file)
        df['experiment_file'] = file
        all_results.append(df)
    
    combined_df = pd.concat(all_results)
    
    # 按配置分组计算平均指标
    summary = combined_df.groupby('experiment_name').agg({
        'context_precision': 'mean',
        'context_recall': 'mean'
    }).reset_index()
    
    # 创建比较图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(summary))
    width = 0.35
    
    ax.bar(x - width/2, summary['context_precision'], width, label='上下文精确度')
    ax.bar(x + width/2, summary['context_recall'], width, label='上下文召回率')
    
    ax.set_xlabel('配置')
    ax.set_ylabel('分数')
    ax.set_title('不同向量数据库配置的性能比较')
    ax.set_xticks(x)
    ax.set_xticklabels(summary['experiment_name'].str.replace('vector_db_test_', ''))
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('vector_db_comparison.png')
    plt.show()
    
    return summary

# 分析结果
experiment_files = [
    "experiments/20241201-vector_db_test_openai_1536.csv",
    "experiments/20241201-vector_db_test_bge_1024.csv",
    "experiments/20241201-vector_db_test_e5_1024.csv"
]

summary = analyze_experiment_results(experiment_files)
print(summary)
```

## 常见问题与解决方案

### Q: 如何选择合适的评估指标？

**A**: 根据您的具体需求选择：
- **检索质量**：使用 Context Precision 和 Context Recall
- **答案质量**：使用 Faithfulness 和 Answer Relevancy
- **整体性能**：结合多个指标获得全面视图

### Q: 测试数据集应该多大？

**A**: 取决于您的具体需求：
- **初步测试**：50-100 个样本
- **详细比较**：200-500 个样本
- **生产评估**：1000+ 个样本

### Q: 如何处理评估中的随机性？

**A**: 
- 使用固定随机种子确保可重现性
- 运行多次评估取平均值
- 报告置信区间或标准差

### Q: 评估成本如何控制？

**A**: 
- 使用较小的模型进行评估（如 gpt-4o-mini）
- 采样评估而非全量评估
- 缓存嵌入和中间结果

## 下一步

1. **扩展评估维度**：添加延迟、吞吐量等性能指标
2. **自动化流程**：集成到 CI/CD 管道
3. **监控生产**：持续监控实际使用中的性能
4. **A/B 测试**：在线上环境中测试改进

## 资源链接

- [Ragas 完整文档](https://docs.ragas.io/)
- [Ragas GitHub](https://github.com/vibrantlabsai/ragas)
- [向量数据库最佳实践](https://www.pinecone.io/learn/vector-database-best-practices)
- [RAG 评估指南](https://docs.ragas.io/en/stable/getstarted/rag_eval.html)