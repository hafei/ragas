# Ragas 向量数据库质量测试指南

## 概述

Ragas 是一个用于评估 LLM 应用程序（特别是 RAG 系统）的框架，它提供了一套系统化的评估方法和指标，帮助您从"感觉检查"转向系统化的评估循环。本指南重点介绍如何使用 Ragas 测试和评估向量数据库中数据集的质量。

## 为什么需要评估向量数据库质量？

向量数据库是 RAG 系统的核心组件，其质量直接影响检索效果和最终答案质量。评估向量数据库质量可以帮助您：

1. **识别检索问题**：发现检索器是否能够找到相关文档
2. **优化嵌入模型**：比较不同嵌入模型在您的数据上的表现
3. **改进分块策略**：确定最佳的文档分块方法
4. **量化性能**：使用客观指标衡量系统改进

## Ragas 核心概念

### 1. 实验驱动方法

Ragas 采用实验驱动的方法，遵循以下循环：

```
进行更改 → 运行评估 → 观察结果 → 提出下一个更改假设
```

### 2. 三大核心组件

- **数据集 (Datasets)**：用于评估的测试数据
- **指标 (Metrics)**：量化性能的度量标准
- **实验 (Experiments)**：系统化的评估流程

## 安装与设置

### 基本安装

```bash
pip install ragas
```

### 开发环境安装

```bash
git clone https://github.com/vibrantlabsai/ragas.git 
pip install -e .
```

### 设置 API 密钥

```bash
export OPENAI_API_KEY="your-openai-key"
# 或其他 LLM 提供商的密钥
```

## 评估向量数据库的关键指标

### 1. 上下文精确度 (Context Precision)

**目的**：评估检索器将相关文档排在前面而非相关文档后面的能力。

**计算方式**：
$$
\text{Context Precision@K} = \frac{\sum_{k=1}^{K} \left( \text{Precision@k} \times v_k \right)}{\text{Total number of relevant items in the top } K \text{ results}}
$$

**使用示例**：
```python
from ragas.metrics.collections import ContextPrecision
from ragas.llms import llm_factory
from openai import AsyncOpenAI

# 设置 LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# 创建指标
scorer = ContextPrecision(llm=llm)

# 评估
result = await scorer.ascore(
    user_input="埃菲尔铁塔位于哪里？",
    reference="埃菲尔铁塔位于巴黎。",
    retrieved_contexts=[
        "埃菲尔铁塔位于巴黎。",
        "勃兰登堡门位于柏林。"
    ]
)
print(f"上下文精确度分数: {result.value}")
```

### 2. 上下文召回率 (Context Recall)

**目的**：衡量成功检索了多少相关文档，专注于不遗漏重要结果。

**计算方式**：
$$
\text{Context Recall} = \frac{\text{Number of claims in the reference supported by the retrieved context}}{\text{Total number of claims in the reference}}
$$

**使用示例**：
```python
from ragas.metrics.collections import ContextRecall

scorer = ContextRecall(llm=llm)

result = await scorer.ascore(
    user_input="埃菲尔铁塔位于哪里？",
    retrieved_contexts=["巴黎是法国的首都。"],
    reference="埃菲尔铁塔位于巴黎。"
)
print(f"上下文召回率分数: {result.value}")
```

### 3. 答案相关性 (Answer Relevancy)

**目的**：衡量回答与用户输入的相关程度，范围从 0 到 1。

**计算方式**：
1. 基于回答生成一组人工问题（默认为 3 个）
2. 计算用户输入嵌入与每个生成问题嵌入之间的余弦相似度
3. 取这些余弦相似度分数的平均值

**使用示例**：
```python
from ragas.metrics.collections import AnswerRelevancy
from ragas.embeddings.base import embedding_factory

scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)

result = await scorer.ascore(
    user_input="第一届超级碗是什么时候举行的？",
    response="第一届超级碗于1967年1月15日举行"
)
print(f"答案相关性分数: {result.value}")
```

### 4. 忠实度 (Faithfulness)

**目的**：衡量回答与检索上下文的事实一致性。

**计算方式**：
$$
\text{Faithfulness Score} = \frac{\text{Number of claims in the response supported by the retrieved context}}{\text{Total number of claims in the response}}
$$

**使用示例**：
```python
from ragas.metrics.collections import Faithfulness

scorer = Faithfulness(llm=llm)

result = await scorer.ascore(
    user_input="爱因斯坦在哪里出生？",
    response="爱因斯坦于1879年3月14日出生于德国。",
    retrieved_contexts=[
        "阿尔伯特·爱因斯坦（1879年3月14日出生）是德国出生的理论物理学家..."
    ]
)
print(f"忠实度分数: {result.value}")
```

## 准备测试数据集

### 1. 使用现有数据集

如果您已有问答对，可以直接转换为 Ragas 数据集格式：

```python
from ragas import Dataset
import pandas as pd

# 创建数据集
dataset = Dataset(name="my_evaluation", backend="local/csv", root_dir="./data")

# 添加样本
data_samples = [
    {
        "question": "什么是 Ragas？",
        "expected_answer": "Ragas 是一个用于评估 LLM 应用程序的库。",
        "metadata": {"complexity": "simple", "category": "introduction"}
    },
    {
        "question": "如何安装 Ragas？",
        "expected_answer": "可以通过 pip install ragas 安装",
        "metadata": {"complexity": "simple", "category": "installation"}
    }
]

for sample in data_samples:
    dataset.append(sample)

dataset.save()
return dataset
```

### 2. 生成合成测试数据

Ragas 提供了强大的测试数据生成功能，可以从您的文档中生成多样化的测试用例：

```python
from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
import openai

# 设置生成器
generator_llm = ChatOpenAI(model="gpt-4o-mini")
critic_llm = ChatOpenAI(model="gpt-4o")
openai_client = openai.OpenAI()
embeddings = OpenAIEmbeddings(client=openai_client)

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# 生成测试集
testset = generator.generate_with_langchain_docs(documents, testset_size=10)
test_df = testset.to_pandas()
```

### 3. 查询类型理解

Ragas 可以生成不同类型的查询：

- **单跳查询 (Single-Hop Query)**：需要从单个文档检索信息的简单问题
- **多跳查询 (Multi-Hop Query)**：需要从多个来源综合信息的复杂问题
- **具体查询 (Specific Query)**：关注具体事实的问题
- **抽象查询 (Abstract Query)**：需要解释或综合的问题

## 向量数据库质量测试流程

### 1. 基础评估流程

```python
from ragas import evaluate
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy

# 定义指标
metrics = [
    ContextPrecision(llm=evaluator_llm),
    ContextRecall(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
    AnswerRelevancy(llm=evaluator_llm, embeddings=embeddings)
]

# 运行评估
result = evaluate(
    dataset=evaluation_dataset,
    metrics=metrics
)

print(result)
```

### 2. 比较不同嵌入模型

```python
# 评估 OpenAI 嵌入
from ragas.embeddings import OpenAIEmbeddings
openai_embeddings = OpenAIEmbeddings(client=openai_client)
query_engine1 = build_query_engine(openai_embeddings)
result1 = evaluate(query_engine1, metrics, test_questions, test_answers)

# 评估 BGE 嵌入
from langchain.embeddings import HuggingFaceEmbeddings
bge_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
query_engine2 = build_query_engine(bge_embeddings)
result2 = evaluate(query_engine2, metrics, test_questions, test_answers)

# 比较结果
print(f"OpenAI 嵌入结果: {result1}")
print(f"BGE 嵌入结果: {result2}")
```

### 3. 实验管理

使用 Ragas 的实验功能进行系统化评估：

```python
from ragas import experiment
import asyncio

@experiment()
async def evaluate_rag_system(row, rag_system, llm):
    """评估 RAG 系统的实验函数"""
    question = row["question"]
    
    # 查询 RAG 系统
    rag_response = await rag_system.query(question)
    model_response = rag_response.get("answer", "")
    
    # 评估忠实度
    faithfulness_score = await Faithfulness(llm=llm).ascore(
        user_input=question,
        response=model_response,
        retrieved_contexts=rag_response.get("retrieved_documents", [])
    )
    
    # 返回评估结果
    return {
        **row,
        "model_response": model_response,
        "faithfulness_score": faithfulness_score.value,
        "retrieved_docs": rag_response.get("retrieved_documents", []),
        "experiment_name": "baseline_v1"
    }

# 运行实验
results = await evaluate_rag_system.arun(
    dataset, 
    rag_system=my_rag_system,
    llm=evaluator_llm
)
```

## 高级评估技术

### 1. 知识图谱方法

Ragas 使用知识图谱方法生成高质量的测试数据：

```python
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.transforms import default_transforms, apply_transforms

# 创建知识图谱
kg = KnowledgeGraph()

# 添加文档节点
for doc in documents:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )

# 应用转换丰富知识图谱
transforms = default_transforms(documents=documents, llm=generator_llm, embedding_model=embeddings)
apply_transforms(kg, transforms)
```

### 2. 场景生成

Ragas 使用基于场景的方法生成多样化的测试用例：

```python
from ragas.testset.synthesizers import default_query_distribution

# 定义查询分布
query_distribution = default_query_distribution(generator_llm)

# 生成测试集
testset = generator.generate(testset_size=10, query_distribution=query_distribution)
```

### 3. 自定义指标

创建适合特定用例的自定义指标：

```python
from ragas.metrics import DiscreteMetric

# 定义正确性指标
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt="""比较模型回答与预期答案，确定是否正确。
    
    如果回答满足以下条件则认为正确：
    1. 包含预期答案的关键信息
    2. 基于提供上下文事实准确
    3. 充分回答了所提问题
    
    返回 'pass' 如果回答正确，否则返回 'fail'。
    
    问题: {question}
    预期答案: {expected_answer}
    模型回答: {response}
    
    评估：""",
    allowed_values=["pass", "fail"],
)
```

## 实际应用案例

### 案例 1：评估文档分块策略

```python
# 测试不同分块大小
chunk_sizes = [256, 512, 1024]
results = {}

for size in chunk_sizes:
    # 使用特定分块大小构建索引
    query_engine = build_query_engine_with_chunk_size(documents, size)
    
    # 评估性能
    result = evaluate(query_engine, metrics, test_questions, test_answers)
    results[f"chunk_size_{size}"] = result

# 比较结果
for size, result in results.items():
    print(f"{size}: Context Precision={result['context_precision']}, Context Recall={result['context_recall']}")
```

### 案例 2：评估检索器类型

```python
# 比较 BM25 与向量检索
from ragas.retrievers import BM25Retriever, VectorRetriever

# BM25 检索器
bm25_retriever = BM25Retriever(documents)
rag_bm25 = RAG(llm_client, bm25_retriever)

# 向量检索器
vector_retriever = VectorRetriever(documents, embeddings)
rag_vector = RAG(llm_client, vector_retriever)

# 评估两种方法
results_bm25 = await evaluate_rag_system.arun(dataset, rag_system=rag_bm25, llm=evaluator_llm)
results_vector = await evaluate_rag_system.arun(dataset, rag_system=rag_vector, llm=evaluator_llm)

# 比较性能
compare_results(results_bm25, results_vector)
```

## 最佳实践

### 1. 数据集准备

- **代表性样本**：确保数据集代表您的 AI 系统将遇到的真实场景
- **平衡分布**：包含不同难度级别、主题和边缘情况的样本
- **质量优于数量**：拥有较少高质量、精心策划的样本比许多低质量样本更好
- **元数据丰富**：包含允许您分析不同维度性能的相关元数据
- **版本控制**：跟踪数据集随时间的变化以确保可重现性

### 2. 实验设计

- **一次更改一个变量**：避免同时进行多个更改，这可能会模糊结果
- **使用描述性名称**：包括更改内容、版本号和日期/时间
- **包含相关元数据**：跟踪模型版本、环境、响应时间等
- **错误处理**：优雅处理错误以避免丢失部分结果

### 3. 指标选择

- **与用例对齐**：选择与您的具体用例相关的简单指标
- **多维度评估**：使用多个指标获得全面视图
- **基线比较**：始终建立基线并与之比较改进
- **持续跟踪**：随时间跟踪指标以识别趋势

## 集成与工作流

### 1. 与 LangChain 集成

```python
from ragas.langchain.evalchain import RagasEvaluatorChain
from langchain.chains import RetrievalQA

# 创建 QA 链
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=index.vectorstore.as_retriever(),
    return_source_documents=True,
)

# 创建评估链
faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
context_precision_chain = RagasEvaluatorChain(metric=context_precision)

# 评估单个结果
result = qa_chain({"query": "你的问题"})
eval_result = faithfulness_chain(result)
```

### 2. 与 LlamaIndex 集成

```python
from ragas.integrations.llama_index import evaluate
from llama_index.core import VectorStoreIndex

# 构建查询引擎
vector_index = VectorStoreIndex.from_documents(documents)
query_engine = vector_index.as_query_engine()

# 评估
result = evaluate(
    query_engine=query_engine,
    metrics=metrics,
    dataset=ragas_dataset,
)
```

### 3. CI/CD 集成

```python
# 在 CI/CD 管道中运行评估
name: rag-evaluation
on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install ragas
        pip install -e .
    - name: Run evaluation
      run: |
        python evals.py
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## 结果分析与改进

### 1. 错误模式分析

检查评估结果 CSV 文件中的失败案例，识别模式：

```python
import pandas as pd

# 加载结果
results_df = pd.read_csv("experiments/experiment_name.csv")

# 分析失败案例
failures = results_df[results_df['correctness_score'] == 'fail']

# 按错误类型分组
error_patterns = failures.groupby('error_type').size()
print(error_patterns)
```

### 2. 系统化改进

基于错误分析，实施针对性改进：

1. **检索问题**：改进分块、尝试混合搜索或更好的嵌入
2. **生成问题**：优化提示、尝试更好的模型
3. **评估问题**：确保指标与用例对齐

### 3. 迭代循环

遵循系统化方法：

1. 创建评估数据集
2. 定义指标
3. 运行基线评估
4. 分析错误模式
5. 实施针对性改进
6. 比较并迭代

## 结论

Ragas 提供了一个全面的框架，用于系统化地评估和改进向量数据库和 RAG 系统的质量。通过采用实验驱动的方法、使用适当的指标和遵循最佳实践，您可以：

- 量化向量数据库性能
- 识别具体的改进领域
- 跟踪随时间的进展
- 做出数据驱动的决策

这种系统化的评估方法使您能够从"感觉检查"转向客观、可重复的评估循环，最终提高您的 RAG 系统的质量和可靠性。

## 资源与进一步阅读

- [Ragas 官方文档](https://docs.ragas.io/)
- [Ragas GitHub 仓库](https://github.com/vibrantlabsai/ragas)
- [Ragas 社区 Discord](https://discord.gg/5djav8GGNZ)
- [示例代码仓库](https://github.com/vibrantlabsai/ragas/tree/main/examples)