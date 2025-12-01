"""
Ragas 向量数据库最终示例

这个示例展示了如何使用 Ragas 进行向量数据库质量评估，
使用最基本的方式，避免复杂的 API 调用。
"""

import os
import pandas as pd

# Ragas 导入
from ragas import evaluate
from ragas.metrics import ContextPrecision, ContextRecall

# LangChain 导入
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


def main():
    """主函数"""
    # 设置 API 密钥
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("请设置 OPENAI_API_KEY 环境变量")
        return
    
    # 初始化 LLM
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
    
    # 初始化嵌入模型
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
        model="text-embedding-3-small"
    )
    
    # 示例文档
    documents = [
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
    
    # 加载文档
    docs = [TextLoader(text=doc).load()[0] for doc in documents]
    
    # 分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(docs)
    
    # 创建向量存储
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # 创建检索器
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # 准备评估数据
    evaluation_data = [
        {
            "user_input": "什么是向量数据库？",
            "retrieved_contexts": [
                "向量数据库是专门用于存储和查询高维向量数据的数据库系统。",
                "它们支持高效的相似性搜索，通常使用近似最近邻算法。"
            ],
            "reference": "向量数据库是专门用于存储和查询高维向量数据的数据库，支持相似性搜索。"
        },
        {
            "user_input": "什么是 HNSW 算法？",
            "retrieved_contexts": [
                "HNSW（Hierarchical Navigable Small World）是一种用于高效近似最近邻搜索的图算法。",
                "它构建多层级的图结构，使得搜索过程能够在对数时间内找到最近邻。"
            ],
            "reference": "HNSW 是一种用于高效近似最近邻搜索的图算法，构建多层级图结构。"
        },
        {
            "user_input": "向量数据库如何提高检索效率？",
            "retrieved_contexts": [
                "向量数据库通过近似最近邻算法如 HNSW、IVF 等提高检索效率。",
                "常见的索引结构有 IVF、PQ、OPQ 等。"
            ],
            "reference": "向量数据库通过近似最近邻算法如 HNSW、IVF 等提高检索效率。"
        }
    ]
    
    # 定义评估指标
    metrics = [
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm)
    ]
    
    # 运行评估
    result = evaluate(dataset=evaluation_data, metrics=metrics)
    
    # 输出结果
    print("向量数据库评估结果:")
    
    # 直接打印结果对象，查看其结构
    print("结果对象类型:", type(result))
    print("结果对象属性:", dir(result))
    
    # 尝试打印结果
    try:
        print("结果:", result)
    except Exception as e:
        print("打印结果时出错:", e)
    
    # 详细分析
    print("\n详细分析:")
    for i, data in enumerate(evaluation_data):
        print(f"\n样本 {i+1}:")
        print(f"  问题: {data['user_input']}")
        print(f"  检索到的上下文数量: {len(data['retrieved_contexts'])}")
        print(f"  参考答案: {data['reference']}")
    
    # 保存结果
    # 创建简单的结果字典
    try:
        # 如果结果是字典，直接使用
        if isinstance(result, dict):
            results_dict = result
        else:
            # 如果是对象，尝试转换为字典
            results_dict = {
                'context_precision': getattr(result, 'context_precision', 'N/A'),
                'context_recall': getattr(result, 'context_recall', 'N/A')
            }
    except Exception as e:
        print("处理结果时出错:", e)
        results_dict = {
            'context_precision': 'N/A',
            'context_recall': 'N/A'
        }
    
    results_df = pd.DataFrame([results_dict])
    results_df.to_csv("vector_db_evaluation_results.csv", index=False)
    print("\n结果已保存到 vector_db_evaluation_results.csv")


if __name__ == "__main__":
    main()