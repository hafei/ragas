"""
Ragas 评估完整示例
展示如何使用修复后的数据集运行 4 个评估指标
"""

import os
import json
from typing import Dict, Any
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas import evaluate
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy
from json_dataset_extractor import JSONDatasetExtractor

def create_complete_dataset_with_responses() -> EvaluationDataset:
    """
    创建包含完整响应的数据集
    这里我们使用更真实的响应数据，而不是简单的期望答案
    """
    
    # 创建数据集提取器
    extractor = JSONDatasetExtractor("test_data.json")
    
    # 加载文档
    documents = extractor.load_documents()
    print(f"加载了 {len(documents)} 个文档")
    
    # 生成查询样本
    query_samples = extractor.generate_query_samples(num_samples=5)
    print(f"生成了 {len(query_samples)} 个查询样本")
    
    # 创建更真实的响应数据
    enhanced_samples = []
    
    for i, query_sample in enumerate(query_samples):
        # 基于上下文生成更自然的响应
        context_text = " ".join(query_sample.context)
        
        # 生成稍微不同的响应，模拟真实的 RAG 系统
        if "向量数据库" in query_sample.question:
            response = f"根据提供的上下文，向量数据库是专门用于存储和查询高维向量数据的数据库系统。它们通过近似最近邻搜索算法（如 HNSW、IVF、LSH 等）来实现高效的向量相似性搜索。"
        elif "Milvus" in query_sample.question:
            response = f"根据上下文信息，Milvus 是一个开源的向量数据库，专为海量特征向量的相似性检索而设计。它支持多种距离度量方式，包括欧氏距离、内积和余弦相似度。"
        elif "HNSW" in query_sample.question:
            response = f"HNSW（Hierarchical Navigable Small World）是一种高效的图索引算法，通过构建多层图结构来实现快速的近似最近邻搜索。它在召回率和搜索速度之间提供了良好的平衡。"
        elif "RAG" in query_sample.question:
            response = f"RAG（Retrieval-Augmented Generation）是一种结合了检索和生成的AI架构。它首先从知识库中检索相关信息，然后基于检索到的内容生成答案，提高了回答的准确性和可靠性。"
        else:
            # 默认使用期望答案
            response = query_sample.expected_answer
        
        # 创建增强的样本
        sample = SingleTurnSample(
            user_input=query_sample.question,
            response=response,  # 使用生成的响应
            retrieved_contexts=query_sample.context,
            reference=query_sample.expected_answer,
            metadata={
                **query_sample.metadata,
                "sample_id": i,
                "response_type": "generated"
            }
        )
        
        enhanced_samples.append(sample)
    
    # 创建评估数据集
    dataset = EvaluationDataset(samples=enhanced_samples)
    print(f"创建了包含 {len(enhanced_samples)} 个样本的增强评估数据集")
    
    return dataset

def run_ragas_evaluation(dataset: EvaluationDataset) -> Dict[str, Any]:
    """
    运行 Ragas 评估
    
    Args:
        dataset: 评估数据集
        
    Returns:
        Dict[str, Any]: 评估结果
    """
    try:
        print("\n=== 开始运行 Ragas 评估 ===")
        
        # 设置环境变量（如果需要）
        if not os.environ.get("OPENAI_API_KEY"):
            print("警告: 未设置 OPENAI_API_KEY，某些评估指标可能无法正常工作")
            # 设置一个虚拟密钥以避免错误
            os.environ["OPENAI_API_KEY"] = "dummy_key_for_testing"
        
        # 定义评估指标
        metrics = [
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy()
        ]
        
        print(f"使用 {len(metrics)} 个评估指标:")
        for metric in metrics:
            print(f"  - {metric.__class__.__name__}")
        
        # 运行评估
        print("\n正在运行评估...")
        result = evaluate(dataset=dataset, metrics=metrics)
        
        # 提取结果
        print("\n=== 评估结果 ===")
        
        # 尝试获取详细结果
        if hasattr(result, 'scores'):
            print("详细评分:")
            for i, sample_scores in enumerate(result.scores):
                print(f"\n样本 {i+1}:")
                for metric, score in sample_scores.items():
                    if isinstance(score, (int, float)):
                        print(f"  {metric}: {score:.4f}")
        
        # 尝试获取汇总结果
        if hasattr(result, '_repr_dict'):
            print("\n汇总结果:")
            summary = result._repr_dict
            for metric, value in summary.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
        else:
            print("\n无法获取汇总结果，显示原始结果:")
            print(result)
        
        return {"success": True, "result": result}
        
    except Exception as e:
        error_msg = f"评估失败: {e}"
        print(f"\n{error_msg}")
        
        # 检查是否是 response 字段相关错误
        if "response" in str(e).lower():
            print("错误提示: 这可能是 response 字段相关的问题")
        
        return {"success": False, "error": str(e)}

def main():
    """主函数"""
    print("Ragas 评估完整示例")
    print("=" * 50)
    
    try:
        # 1. 创建包含响应的数据集
        dataset = create_complete_dataset_with_responses()
        
        # 2. 运行评估
        evaluation_result = run_ragas_evaluation(dataset)
        
        # 3. 输出最终结果
        print("\n" + "=" * 50)
        print("最终结果:")
        
        if evaluation_result["success"]:
            print("[SUCCESS] Ragas 评估成功完成！")
            print("数据集已正确包含 response 字段，所有 4 个评估指标都可以正常运行。")
        else:
            print("[FAILED] 评估失败")
            print(f"错误信息: {evaluation_result['error']}")
            
            # 提供解决建议
            if "response" in evaluation_result['error'].lower():
                print("\n解决建议:")
                print("1. 确保数据集包含 response 字段")
                print("2. 检查 response 字段是否为空")
                print("3. 验证数据集格式是否正确")
        
    except Exception as e:
        print(f"\n程序执行失败: {e}")
        print("请检查配置和数据文件是否存在")

if __name__ == "__main__":
    main()