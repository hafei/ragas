"""
Ragas 向量数据库质量测试简化示例

这个示例展示了如何使用 Ragas 评估向量数据库中数据集的质量，
包括不同嵌入模型、分块策略和检索器类型的比较。
"""

import os
import asyncio
import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt

# Ragas 导入
from ragas import Dataset, evaluate
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy

# LangChain 导入
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

# OpenAI 导入
import openai


class SimpleVectorDBEvaluator:
    """简化的向量数据库质量评估器"""
    
    def __init__(self, openai_api_key: str):
        """初始化评估器"""
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.evaluator_llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
        
        # 初始化嵌入模型
        self.openai_embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small"
        )
        self.bge_embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
    def create_sample_dataset(self, output_path: str = "sample_dataset.csv") -> Dataset:
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
    
    def build_vector_store(self, documents: List[str], embedding_model, chunk_size: int = 512, chunk_overlap: int = 50):
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
    
    async def evaluate_vector_store(self, vector_store, testset: Dataset, store_name: str) -> Dict[str, float]:
        """评估单个向量存储"""
        # 创建检索器
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        # 评估每个样本
        results = []
        test_df = testset.to_pandas()
        
        for _, row in test_df.iterrows():
            question = row["question"]
            reference = row["reference"]
            
            # 检索相关文档
            retrieved_docs = retriever.get_relevant_documents(question)
            retrieved_contexts = [doc.page_content for doc in retrieved_docs]
            
            # 生成回答（简单模拟）
            response = f"基于检索到的文档，{reference}"
            
            # 评估指标
            context_precision = ContextPrecision(llm=self.evaluator_llm)
            context_recall = ContextRecall(llm=self.evaluator_llm)
            faithfulness = Faithfulness(llm=self.evaluator_llm)
            answer_relevancy = AnswerRelevancy(llm=self.evaluator_llm, embeddings=self.openai_embeddings)
            
            # 计算分数
            cp_score = await context_precision.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=retrieved_contexts
            )
            
            cr_score = await context_recall.ascore(
                user_input=question,
                retrieved_contexts=retrieved_contexts,
                reference=reference
            )
            
            f_score = await faithfulness.ascore(
                user_input=question,
                response=response,
                retrieved_contexts=retrieved_contexts
            )
            
            ar_score = await answer_relevancy.ascore(
                user_input=question,
                response=response
            )
            
            results.append({
                "question": question,
                "reference": reference,
                "retrieved_count": len(retrieved_contexts),
                "context_precision": cp_score.value,
                "context_recall": cr_score.value,
                "faithfulness": f_score.value,
                "answer_relevancy": ar_score.value
            })
        
        # 计算平均分数
        df = pd.DataFrame(results)
        avg_scores = {
            "context_precision": df["context_precision"].mean(),
            "context_recall": df["context_recall"].mean(),
            "faithfulness": df["faithfulness"].mean(),
            "answer_relevancy": df["answer_relevancy"].mean()
        }
        
        print(f"{store_name} 平均分数:")
        print(f"  上下文精确度: {avg_scores['context_precision']:.3f}")
        print(f"  上下文召回率: {avg_scores['context_recall']:.3f}")
        print(f"  忠实度: {avg_scores['faithfulness']:.3f}")
        print(f"  答案相关性: {avg_scores['answer_relevancy']:.3f}")
        
        return avg_scores
    
    async def evaluate_embedding_models(self, documents: List[str], testset: Dataset):
        """比较不同嵌入模型的性能"""
        print("评估不同嵌入模型...")
        
        results = {}
        
        # 评估 OpenAI 嵌入
        print("评估 OpenAI 嵌入...")
        openai_store = self.build_vector_store(documents, self.openai_embeddings)
        openai_result = await self.evaluate_vector_store(openai_store, testset, "OpenAI")
        results["OpenAI"] = openai_result
        
        # 评估 BGE 嵌入
        print("评估 BGE 嵌入...")
        bge_store = self.build_vector_store(documents, self.bge_embeddings)
        bge_result = await self.evaluate_vector_store(bge_store, testset, "BGE")
        results["BGE"] = bge_result
        
        return results
    
    async def evaluate_chunking_strategies(self, documents: List[str], testset: Dataset):
        """比较不同分块策略的性能"""
        print("评估不同分块策略...")
        
        chunk_strategies = [
            {"name": "small_chunks", "chunk_size": 256, "chunk_overlap": 25},
            {"name": "medium_chunks", "chunk_size": 512, "chunk_overlap": 50},
            {"name": "large_chunks", "chunk_size": 1024, "chunk_overlap": 100}
        ]
        
        results = {}
        
        for strategy in chunk_strategies:
            print(f"评估 {strategy['name']} 策略...")
            vector_store = self.build_vector_store(
                documents, 
                self.openai_embeddings,
                strategy["chunk_size"],
                strategy["chunk_overlap"]
            )
            result = await self.evaluate_vector_store(vector_store, testset, strategy["name"])
            results[strategy["name"]] = result
        
        return results
    
    def visualize_results(self, results: Dict[str, Dict[str, float]], title: str = "向量数据库性能比较"):
        """可视化评估结果"""
        # 准备数据
        models = list(results.keys())
        metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        for i, metric in enumerate(metrics):
            row, col = i // 2, i % 2
            values = [results[model][metric] for model in models]
            
            axes[row, col].bar(models, values)
            axes[row, col].set_title(f"{metric.replace('_', ' ').title()}")
            axes[row, col].set_ylabel('分数')
            axes[row, col].set_ylim(0, 1)
            
            # 添加数值标签
            for j, v in enumerate(values):
                axes[row, col].text(j, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(f"{title.lower().replace(' ', '_')}.png")
        plt.show()
    
    async def run_comprehensive_evaluation(self, documents: List[str]):
        """运行全面的向量数据库评估"""
        print("开始全面向量数据库评估...")
        
        # 创建测试数据集
        testset = self.create_sample_dataset()
        
        # 评估嵌入模型
        embedding_results = await self.evaluate_embedding_models(documents, testset)
        
        # 评估分块策略
        chunking_results = await self.evaluate_chunking_strategies(documents, testset)
        
        # 可视化结果
        self.visualize_results(embedding_results, "不同嵌入模型性能比较")
        self.visualize_results(chunking_results, "不同分块策略性能比较")
        
        return {
            "embedding_results": embedding_results,
            "chunking_results": chunking_results
        }


async def main():
    """主函数"""
    # 设置 API 密钥
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("请设置 OPENAI_API_KEY 环境变量")
        return
    
    # 创建评估器
    evaluator = SimpleVectorDBEvaluator(openai_api_key)
    
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
    
    # 运行全面评估
    results = await evaluator.run_comprehensive_evaluation(sample_documents)
    
    print("\n评估完成！")
    print("结果图表已保存为 PNG 文件。")


if __name__ == "__main__":
    asyncio.run(main())