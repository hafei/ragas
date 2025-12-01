"""
Ragas 向量数据库基础评估示例

这个示例展示了如何使用 Ragas 进行基础的向量数据库质量评估，
包括上下文精确度和上下文召回率的计算。
"""

import os
import pandas as pd
from typing import List, Dict

# Ragas 导入
from ragas import Dataset, evaluate
from ragas.metrics import ContextPrecision, ContextRecall

# LangChain 导入
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# OpenAI 导入
import openai


def create_sample_dataset() -> Dataset:
    """创建示例评估数据集"""
    dataset = Dataset(name="vector_db_eval", backend="local/csv", root_dir=".")
    
    # 创建示例问答对
    samples = [
        {
            "question": "什么是向量数据库？",
            "reference": "向量数据库是专门用于存储和查询高维向量数据的数据库，支持相似性搜索。",
            "metadata": {"category": "concept", "difficulty": "easy"}
        },
        {
            "question": "向量数据库如何提高检索效率？",
            "reference": "向量数据库通过近似最近邻算法如 HNSW、IVF 等提高检索效率。",
            "metadata": {"category": "technical", "difficulty": "medium"}
        },
        {
            "question": "什么是 HNSW 算法？",
            "reference": "HNSW（Hierarchical Navigable Small World）是一种用于高效近似最近邻搜索的图算法。",
            "metadata": {"category": "algorithm", "difficulty": "hard"}
        }
    ]
    
    for sample in samples:
        dataset.append(sample)
    
    dataset.save()
    return dataset


def build_vector_store(documents: List[str], embedding_model, chunk_size: int = 512, chunk_overlap: int = 50):
    """构建向量存储"""
    # 加载文档
    docs = [TextLoader(text=doc).load()[0] for doc in documents]
    
    # 分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    
    # 创建向量存储
    vector_store = FAISS.from_documents(chunks, embedding_model)
    
    return vector_store


def evaluate_vector_store(vector_store, testset: Dataset, store_name: str, evaluator_llm) -> Dict[str, float]:
    """评估单个向量存储"""
    # 定义评估指标
    metrics = [
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm)
    ]
    
    # 创建检索器
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # 准备评估数据
    evaluation_data = []
    test_df = testset.to_pandas()
    
    for _, row in test_df.iterrows():
        question = row["question"]
        reference = row["reference"]
        
        # 检索相关文档
        retrieved_docs = retriever.get_relevant_documents(question)
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]
        
        # 生成回答（简单模拟）
        response = f"基于检索到的文档，{reference}"
        
        # 准备评估样本
        sample = {
            "user_input": question,
            "retrieved_contexts": retrieved_contexts,
            "reference": reference,
            "response": response
        }
        
        evaluation_data.append(sample)
    
    # 创建评估数据集
    eval_dataset = Dataset.from_list(evaluation_data)
    
    # 运行评估
    result = evaluate(dataset=eval_dataset, metrics=metrics)
    
    print(f"{store_name} 评估结果:")
    print(f"  上下文精确度: {result['context_precision']:.3f}")
    print(f"  上下文召回率: {result['context_recall']:.3f}")
    
    return result


def compare_embedding_models(documents: List[str], testset: Dataset, openai_api_key: str):
    """比较不同嵌入模型的性能"""
    print("比较不同嵌入模型...")
    
    # 初始化 LLM
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
    
    results = {}
    
    # 评估 OpenAI 嵌入
    print("评估 OpenAI 嵌入...")
    openai_embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
        model="text-embedding-3-small"
    )
    openai_store = build_vector_store(documents, openai_embeddings)
    openai_result = evaluate_vector_store(openai_store, testset, "OpenAI", evaluator_llm)
    results["OpenAI"] = openai_result
    
    # 评估 BGE 嵌入（如果可用）
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        print("评估 BGE 嵌入...")
        bge_embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
        bge_store = build_vector_store(documents, bge_embeddings)
        bge_result = evaluate_vector_store(bge_store, testset, "BGE", evaluator_llm)
        results["BGE"] = bge_result
    except Exception as e:
        print(f"BGE 嵌入评估失败: {e}")
    
    return results


def compare_chunking_strategies(documents: List[str], testset: Dataset, openai_api_key: str):
    """比较不同分块策略的性能"""
    print("比较不同分块策略...")
    
    # 初始化 LLM
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
    openai_embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
        model="text-embedding-3-small"
    )
    
    chunk_strategies = [
        {"name": "small_chunks", "chunk_size": 256, "chunk_overlap": 25},
        {"name": "medium_chunks", "chunk_size": 512, "chunk_overlap": 50},
        {"name": "large_chunks", "chunk_size": 1024, "chunk_overlap": 100}
    ]
    
    results = {}
    
    for strategy in chunk_strategies:
        print(f"评估 {strategy['name']} 策略...")
        vector_store = build_vector_store(
            documents, 
            openai_embeddings,
            strategy["chunk_size"],
            strategy["chunk_overlap"]
        )
        result = evaluate_vector_store(vector_store, testset, strategy["name"], evaluator_llm)
        results[strategy["name"]] = result
    
    return results


def visualize_results(results: Dict[str, Dict[str, float]], title: str = "向量数据库性能比较"):
    """可视化评估结果"""
    # 准备数据
    models = list(results.keys())
    metrics = ["context_precision", "context_recall"]
    
    # 创建图表
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=16)
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        
        axes[i].bar(models, values)
        axes[i].set_title(f"{metric.replace('_', ' ').title()}")
        axes[i].set_ylabel('分数')
        axes[i].set_ylim(0, 1)
        
        # 添加数值标签
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.show()


def main():
    """主函数"""
    # 设置 API 密钥
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("请设置 OPENAI_API_KEY 环境变量")
        return
    
    # 示例文档
    sample_documents = [
        """
        向量数据库是专门用于存储和查询高维向量数据的数据库系统。它们支持高效的相似性搜索，
        通常使用近似最近邻算法如 HNSW、IVF、LSH 等。向量数据库广泛应用于
        推荐系统、图像搜索、自然语言处理等领域。
        """,
        """
        HNSW（Hierarchical Navigable Small World）是一种用于高效近似最近邻搜索的图算法。
        它构建多层级的图结构，使得搜索过程能够在对数时间内找到最近邻。
        HNSW 在向量数据库中被广泛采用，特别是在需要高吞吐量的场景中。
        """,
        """
        向量数据库的性能优化包括多个方面：索引结构选择、参数调优、
        硬件加速等。常见的索引结构有 IVF（倒排文件）、PQ（乘积量化）、
        OPQ（优化乘积量化）等。选择合适的索引结构对查询性能至关重要。
        """
    ]
    
    # 创建测试数据集
    testset = create_sample_dataset()
    
    # 评估嵌入模型
    embedding_results = compare_embedding_models(sample_documents, testset, openai_api_key)
    
    # 评估分块策略
    chunking_results = compare_chunking_strategies(sample_documents, testset, openai_api_key)
    
    # 可视化结果
    visualize_results(embedding_results, "不同嵌入模型性能比较")
    visualize_results(chunking_results, "不同分块策略性能比较")
    
    print("\n评估完成！")
    print("结果图表已保存为 PNG 文件。")


if __name__ == "__main__":
    main()