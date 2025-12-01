"""
测试 Ragas 评估修复
验证 response 字段生成和四个评估指标是否正常工作
"""

import os
import json
import sys
from typing import Dict, Any, List

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ragas_siliconflow_milvus_test import RagasSiliconFlowMilvusTest
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy


def create_minimal_test_dataset() -> EvaluationDataset:
    """创建一个最小的测试数据集，包含所有必需字段"""
    
    # 创建测试样本
    samples = [
        SingleTurnSample(
            user_input="什么是向量数据库？",
            response="向量数据库是专门用于存储和查询高维向量数据的数据库系统。它们通过近似最近邻搜索算法来实现高效的向量相似性搜索。",
            retrieved_contexts=[
                "向量数据库是专门用于存储和查询高维向量数据的数据库系统。",
                "它们通过近似最近邻搜索算法（如 HNSW、IVF、LSH 等）来实现高效的向量相似性搜索。"
            ],
            reference="向量数据库是专门用于存储和查询高维向量数据的数据库，支持相似性搜索。"
        ),
        SingleTurnSample(
            user_input="什么是 HNSW 算法？",
            response="HNSW（Hierarchical Navigable Small World）是一种高效的图索引算法，通过构建多层图结构来实现快速的近似最近邻搜索。",
            retrieved_contexts=[
                "HNSW（Hierarchical Navigable Small World）是一种高效的图索引算法。",
                "它通过构建多层图结构来实现快速的近似最近邻搜索。"
            ],
            reference="HNSW 是一种用于高效近似最近邻搜索的图算法。"
        )
    ]
    
    return EvaluationDataset(samples=samples)


def test_minimal_evaluation():
    """测试最小评估数据集"""
    print("=== 测试最小评估数据集 ===")
    
    try:
        # 创建测试数据集
        dataset = create_minimal_test_dataset()
        print(f"[OK] 成功创建包含 {len(dataset.samples)} 个样本的测试数据集")
        
        # 验证每个样本的字段
        for i, sample in enumerate(dataset.samples):
            print(f"\n样本 {i+1}:")
            print(f"  user_input: {sample.user_input}")
            print(f"  response: {sample.response[:50]}...")
            contexts_count = len(sample.retrieved_contexts) if sample.retrieved_contexts else 0
            print(f"  retrieved_contexts: {contexts_count} 个上下文")
            print(f"  reference: {sample.reference[:50]}...")
            
            # 检查必需字段
            assert sample.user_input is not None, "user_input 不能为空"
            assert sample.response is not None, "response 不能为空"
            assert sample.retrieved_contexts is not None, "retrieved_contexts 不能为空"
            assert sample.reference is not None, "reference 不能为空"
        
        print("\n[OK] 所有样本字段验证通过")
        return True
        
    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_response_generation():
    """测试响应生成功能"""
    print("\n=== 测试响应生成功能 ===")
    
    try:
        # 加载配置
        config_path = "config.json"
        if not os.path.exists(config_path):
            print("[ERROR] 配置文件不存在，跳过响应生成测试")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 创建测试实例
        test = RagasSiliconFlowMilvusTest(config)
        
        # 初始化组件（只初始化 LLM）
        if config.get("llm_api_key"):
            from openai import AsyncOpenAI
            from ragas.llms import llm_factory
            
            client = AsyncOpenAI(
                api_key=config["llm_api_key"],
                base_url=config.get("llm_base_url", "https://api.openai.com/v1")
            )
            
            test.evaluator_llm = llm_factory(
                config.get("evaluator_model", "gpt-4o-mini"),
                client=client
            )
            print("[OK] LLM 初始化成功")
        else:
            print("[ERROR] 未配置 LLM API 密钥")
            return False
        
        # 测试响应生成
        question = "什么是向量数据库？"
        contexts = [
            "向量数据库是专门用于存储和查询高维向量数据的数据库系统。",
            "它们通过近似最近邻搜索算法来实现高效的向量相似性搜索。"
        ]
        
        response = test._generate_response_from_context(question, contexts)
        print(f"[OK] 响应生成成功: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 响应生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_metrics():
    """测试评估指标"""
    print("\n=== 测试评估指标 ===")
    
    try:
        # 加载配置
        config_path = "config.json"
        if not os.path.exists(config_path):
            print("[ERROR] 配置文件不存在，跳过评估指标测试")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 创建测试实例
        test = RagasSiliconFlowMilvusTest(config)
        
        # 初始化 LLM
        if config.get("llm_api_key"):
            from openai import AsyncOpenAI
            from ragas.llms import llm_factory
            
            client = AsyncOpenAI(
                api_key=config["llm_api_key"],
                base_url=config.get("llm_base_url", "https://api.openai.com/v1")
            )
            
            test.evaluator_llm = llm_factory(
                config.get("evaluator_model", "gpt-4o-mini"),
                client=client
            )
            print("[OK] LLM 初始化成功")
        else:
            print("[ERROR] 未配置 LLM API 密钥")
            return False
        
        # 创建测试数据集
        dataset = create_minimal_test_dataset()
        
        # 初始化嵌入模型
        from siliconflow_embeddings import SiliconFlowEmbeddings
        embeddings = SiliconFlowEmbeddings(
            api_key=config["siliconflow_api_key"],
            model_name=config.get("embedding_model", "BAAI/bge-large-zh-v1.5"),
            base_url=config.get("embedding_base_url", "https://api.siliconflow.cn/v1")
        )
        
        # 测试每个指标
        metrics = [
            ("ContextPrecision", ContextPrecision(llm=test.evaluator_llm)),
            ("ContextRecall", ContextRecall(llm=test.evaluator_llm)),
            ("Faithfulness", Faithfulness(llm=test.evaluator_llm)),
            ("AnswerRelevancy", AnswerRelevancy(llm=test.evaluator_llm, embeddings=embeddings))
        ]
        
        print("\n测试各个评估指标:")
        
        for metric_name, metric in metrics:
            try:
                print(f"  测试 {metric_name}...")
                
                # 使用第一个样本进行测试
                sample = dataset.samples[0]
                
                # 根据指标类型调用不同的评分方法
                import asyncio
                
                # 创建事件循环
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # 调用评分方法
                if metric_name == "ContextPrecision":
                    score = loop.run_until_complete(
                        metric.single_turn_ascore(sample)
                    )
                elif metric_name == "ContextRecall":
                    score = loop.run_until_complete(
                        metric.single_turn_ascore(sample)
                    )
                elif metric_name == "Faithfulness":
                    score = loop.run_until_complete(
                        metric.single_turn_ascore(sample)
                    )
                elif metric_name == "AnswerRelevancy":
                    score = loop.run_until_complete(
                        metric.single_turn_ascore(sample)
                    )
                
                print(f"    [OK] {metric_name}: {score:.4f}")
                
            except Exception as e:
                print(f"    [ERROR] {metric_name} 测试失败: {e}")
                return False
        
        print("\n[OK] 所有评估指标测试通过")
        return True
        
    except Exception as e:
        print(f"[ERROR] 评估指标测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_complete_evaluation():
    """测试完整评估流程"""
    print("\n=== 测试完整评估流程 ===")
    
    try:
        # 加载配置
        config_path = "config.json"
        if not os.path.exists(config_path):
            print("[ERROR] 配置文件不存在，跳过完整评估测试")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 创建测试实例
        test = RagasSiliconFlowMilvusTest(config)
        
        # 初始化 LLM
        if config.get("llm_api_key"):
            from openai import AsyncOpenAI
            from ragas.llms import llm_factory
            
            client = AsyncOpenAI(
                api_key=config["llm_api_key"],
                base_url=config.get("llm_base_url", "https://api.openai.com/v1")
            )
            
            test.evaluator_llm = llm_factory(
                config.get("evaluator_model", "gpt-4o-mini"),
                client=client
            )
            print("[OK] LLM 初始化成功")
        else:
            print("[ERROR] 未配置 LLM API 密钥")
            return False
        
        # 创建测试数据集
        dataset = create_minimal_test_dataset()
        
        # 运行评估
        result = test.run_ragas_evaluation(dataset)
        
        if "error" in result:
            print(f"[ERROR] 评估失败: {result['error']}")
            return False
        
        print("[OK] 完整评估流程成功")
        print(f"评估结果: {result}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 完整评估测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("Ragas 评估修复验证测试")
    print("=" * 50)
    
    # 运行测试
    tests = [
        ("最小评估数据集", test_minimal_evaluation),
        ("响应生成功能", test_response_generation),
        ("评估指标", test_evaluation_metrics),
        ("完整评估流程", test_complete_evaluation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        success = test_func()
        results.append((test_name, success))
    
    # 输出测试结果摘要
    print("\n" + "=" * 50)
    print("测试结果摘要:")
    for test_name, success in results:
        status = "[PASS] 通过" if success else "[FAIL] 失败"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n[SUCCESS] 所有测试都通过了！Ragas 评估修复成功。")
        print("\n下一步:")
        print("1. 运行完整的测试: python ragas_siliconflow_milvus_test.py")
        print("2. 验证所有四个评估指标都能正常运行")
    else:
        print("\n[WARNING] 部分测试失败，请检查错误信息。")


if __name__ == "__main__":
    main()