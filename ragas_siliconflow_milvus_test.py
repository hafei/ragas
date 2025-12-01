"""
Ragas + SiliconFlow + Milvus 完整测试脚本
整合所有组件的端到端测试
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional

# 导入自定义组件
from siliconflow_embeddings import SiliconFlowEmbeddings
from milvus_connector import MilvusConnector
from json_dataset_extractor import JSONDatasetExtractor

# 导入 Ragas 组件
from ragas import evaluate
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy
from ragas.llms import llm_factory, BaseRagasLLM, InstructorBaseRagasLLM
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample


class RagasSiliconFlowMilvusTest:
    """
    Ragas + SiliconFlow + Milvus 集成测试类
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化测试类
        
        Args:
            config: 配置字典，包含 API 密钥、数据库连接等信息
        """
        self.config = config
        self.embeddings = None
        self.milvus = None
        self.dataset_extractor = None
        self.evaluator_llm = None
        
    def setup(self) -> bool:
        """设置所有组件"""
        try:
            print("=== 初始化组件 ===")
            
            # 1. 初始化 SiliconFlow 嵌入模型
            print("1. 初始化 SiliconFlow 嵌入模型...")
            self.embeddings = SiliconFlowEmbeddings(
                api_key=self.config["siliconflow_api_key"],
                model_name=self.config.get("embedding_model", "BAAI/bge-large-zh-v1.5")
            )
            print("✓ SiliconFlow 嵌入模型初始化成功")
            
            # 2. 初始化 Milvus 连接器
            print("2. 初始化 Milvus 连接器...")
            self.milvus = MilvusConnector(
                host=self.config.get("milvus_host", "localhost"),
                port=self.config.get("milvus_port", 19530),
                user=self.config.get("milvus_user"),
                password=self.config.get("milvus_password"),
                collection_name=self.config.get("milvus_collection", "ragas_test_docs")
            )
            
            if not self.milvus.connect():
                raise Exception("Milvus 连接失败")
            print("✓ Milvus 连接成功")
            
            # 3. 初始化数据集提取器
            print("3. 初始化数据集提取器...")
            self.dataset_extractor = JSONDatasetExtractor(
                self.config["json_data_path"]
            )
            print("✓ 数据集提取器初始化成功")
            
            # 4. 初始化评估 LLM（使用 OpenAI 兼容的供应商）
            print("4. 初始化评估 LLM...")
            if self.config.get("llm_api_key"):
                from openai import AsyncOpenAI
                
                # 使用 OpenAI 兼容的供应商配置
                client = AsyncOpenAI(
                    api_key=self.config["llm_api_key"],
                    base_url=self.config.get("llm_base_url", "https://api.openai.com/v1")
                )
                
                self.evaluator_llm = llm_factory(
                    self.config.get("evaluator_model", "gpt-4o-mini"),
                    client=client
                )
                
                provider_name = self.config.get("llm_provider", "OpenAI")
                print(f"✓ {provider_name} 评估 LLM 初始化成功")
            else:
                print("⚠ 未提供 LLM API 密钥，将使用模拟评估")
                self.evaluator_llm = None
            
            return True
            
        except Exception as e:
            print(f"✗ 初始化失败: {e}")
            return False
    
    def setup_milvus_collection(self) -> bool:
        """设置 Milvus 集合"""
        try:
            print("\n=== 设置 Milvus 集合 ===")
            
            # 获取嵌入维度
            dimension = self.embeddings.get_embedding_dimension()
            print(f"嵌入向量维度: {dimension}")
            
            # 创建集合
            if not self.milvus.create_collection(dimension):
                raise Exception("创建 Milvus 集合失败")
            print("✓ Milvus 集合创建成功")
            
            # 创建索引
            if not self.milvus.create_index():
                raise Exception("创建 Milvus 索引失败")
            print("✓ Milvus 索引创建成功")
            
            return True
            
        except Exception as e:
            print(f"✗ Milvus 集合设置失败: {e}")
            return False
    
    def load_documents_to_milvus(self) -> bool:
        """加载文档到 Milvus"""
        try:
            print("\n=== 加载文档到 Milvus ===")
            
            # 加载文档
            documents = self.dataset_extractor.load_documents()
            if not documents:
                raise Exception("没有找到文档")
            
            print(f"加载了 {len(documents)} 个文档")
            
            # 转换文档格式
            milvus_docs = []
            for doc in documents:
                milvus_docs.append({
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata
                })
            
            # 插入到 Milvus
            if not self.milvus.insert_documents(milvus_docs, self.embeddings):
                raise Exception("插入文档到 Milvus 失败")
            
            print(f"✓ 成功插入 {len(milvus_docs)} 个文档到 Milvus")
            
            # 显示集合统计
            stats = self.milvus.get_collection_stats()
            print(f"集合统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")
            
            return True
            
        except Exception as e:
            print(f"✗ 加载文档失败: {e}")
            return False
    
    def create_evaluation_dataset(self) -> EvaluationDataset:
        """创建评估数据集"""
        try:
            print("\n=== 创建评估数据集 ===")
            
            # 生成查询样本
            query_samples = self.dataset_extractor.generate_query_samples(
                num_samples=self.config.get("num_samples", 10)
            )
            
            print(f"生成了 {len(query_samples)} 个查询样本")
            
            # 创建 Ragas 数据集
            dataset = self.dataset_extractor.create_ragas_dataset()
            
            print(f"✓ 创建了包含 {len(dataset.samples)} 个样本的评估数据集")
            
            return dataset
            
        except Exception as e:
            print(f"✗ 创建评估数据集失败: {e}")
            return None
    
    def test_milvus_search(self) -> bool:
        """测试 Milvus 搜索功能"""
        try:
            print("\n=== 测试 Milvus 搜索 ===")
            
            test_queries = [
                "什么是向量数据库？",
                "Milvus 有什么特点？",
                "HNSW 算法如何工作？"
            ]
            
            for query in test_queries:
                print(f"\n查询: {query}")
                results = self.milvus.search(query, self.embeddings, top_k=3)
                
                if results:
                    print(f"找到 {len(results)} 个相关文档:")
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. [分数: {result['score']:.4f}] {result['content'][:50]}...")
                else:
                    print("  未找到相关文档")
            
            print("✓ Milvus 搜索测试完成")
            return True
            
        except Exception as e:
            print(f"✗ Milvus 搜索测试失败: {e}")
            return False
    
    def run_ragas_evaluation(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        """运行 Ragas 评估"""
        try:
            print("\n=== 运行 Ragas 评估 ===")
            
            if not self.evaluator_llm:
                print("⚠ 未配置评估 LLM，跳过评估")
                return {"error": "未配置评估 LLM"}
            
            # 定义评估指标
            metrics = [
                ContextPrecision(llm=self.evaluator_llm),
                ContextRecall(llm=self.evaluator_llm),
                Faithfulness(llm=self.evaluator_llm),
                AnswerRelevancy(llm=self.evaluator_llm)
            ]
            
            print(f"使用 {len(metrics)} 个评估指标")
            
            # 运行评估
            result = evaluate(dataset=dataset, metrics=metrics)
            
            print("✓ 评估完成")
            print(f"评估结果: {result}")
            
            return result
            
        except Exception as e:
            print(f"✗ Ragas 评估失败: {e}")
            return {"error": str(e)}
    
    def create_retrieval_evaluation_dataset(self) -> EvaluationDataset:
        """创建基于检索的评估数据集"""
        try:
            print("\n=== 创建基于检索的评估数据集 ===")
            
            # 获取原始查询样本
            query_samples = self.dataset_extractor.query_samples
            
            retrieval_samples = []
            
            for query_sample in query_samples:
                # 使用 Milvus 检索相关文档
                retrieved_docs = self.milvus.search(
                    query_sample.question,
                    self.embeddings,
                    top_k=5
                )
                
                # 提取检索到的内容
                retrieved_contexts = [doc["content"] for doc in retrieved_docs]
                
                # 创建 Ragas 样本
                sample = SingleTurnSample(
                    user_input=query_sample.question,
                    retrieved_contexts=retrieved_contexts,
                    reference=query_sample.expected_answer,
                    metadata={
                        "original_context": query_sample.context,
                        "retrieval_scores": [doc["score"] for doc in retrieved_docs]
                    }
                )
                
                retrieval_samples.append(sample)
            
            # 创建评估数据集
            retrieval_dataset = EvaluationDataset(samples=retrieval_samples)
            
            print(f"✓ 创建了包含 {len(retrieval_samples)} 个样本的检索评估数据集")
            
            return retrieval_dataset
            
        except Exception as e:
            print(f"✗ 创建检索评估数据集失败: {e}")
            return None
    
    def run_complete_test(self) -> Dict[str, Any]:
        """运行完整测试流程"""
        results = {
            "setup_success": False,
            "milvus_setup_success": False,
            "documents_loaded": False,
            "search_test_success": False,
            "evaluation_results": None,
            "retrieval_evaluation_results": None,
            "errors": []
        }
        
        try:
            # 1. 设置组件
            if not self.setup():
                return results
            results["setup_success"] = True
            
            # 2. 设置 Milvus 集合
            if not self.setup_milvus_collection():
                return results
            results["milvus_setup_success"] = True
            
            # 3. 加载文档到 Milvus
            if not self.load_documents_to_milvus():
                return results
            results["documents_loaded"] = True
            
            # 4. 测试搜索功能
            if not self.test_milvus_search():
                return results
            results["search_test_success"] = True
            
            # 5. 创建评估数据集
            dataset = self.create_evaluation_dataset()
            if not dataset:
                return results
            
            # 6. 运行基础评估
            evaluation_results = self.run_ragas_evaluation(dataset)
            results["evaluation_results"] = evaluation_results
            
            # 7. 创建基于检索的评估数据集
            retrieval_dataset = self.create_retrieval_evaluation_dataset()
            if retrieval_dataset:
                # 8. 运行检索评估
                retrieval_evaluation_results = self.run_ragas_evaluation(retrieval_dataset)
                results["retrieval_evaluation_results"] = retrieval_evaluation_results
            
            print("\n=== 测试完成 ===")
            print("所有测试步骤已成功完成！")
            
        except Exception as e:
            error_msg = f"测试过程中发生错误: {e}"
            print(f"✗ {error_msg}")
            results["errors"].append(error_msg)
        
        finally:
            # 清理资源
            if self.milvus:
                self.milvus.disconnect()
        
        return results


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """加载配置文件"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 返回默认配置
            return {
                "siliconflow_api_key": os.getenv("SILICONFLOW_API_KEY", ""),
                "llm_provider": os.getenv("LLM_PROVIDER", "siliconflow"),
                "llm_base_url": os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1"),
                "llm_api_key": os.getenv("LLM_API_KEY", os.getenv("SILICONFLOW_API_KEY", "")),
                "json_data_path": "test_data.json",
                "embedding_model": "BAAI/bge-large-zh-v1.5",
                "evaluator_model": os.getenv("EVALUATOR_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
                "milvus_host": "localhost",
                "milvus_port": 19530,
                "milvus_collection": "ragas_test_docs",
                "num_samples": 10
            }
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}


def main():
    """主函数"""
    print("Ragas + SiliconFlow + Milvus 集成测试")
    print("=" * 50)
    
    # 加载配置
    config = load_config()
    
    # 验证必需的配置
    if not config.get("siliconflow_api_key"):
        print("错误: 请设置 SILICONFLOW_API_KEY 环境变量或在配置文件中提供")
        return
    
    # 创建测试实例
    test = RagasSiliconFlowMilvusTest(config)
    
    # 运行完整测试
    results = test.run_complete_test()
    
    # 输出结果摘要
    print("\n" + "=" * 50)
    print("测试结果摘要:")
    print(f"组件设置: {'✓' if results['setup_success'] else '✗'}")
    print(f"Milvus 设置: {'✓' if results['milvus_setup_success'] else '✗'}")
    print(f"文档加载: {'✓' if results['documents_loaded'] else '✗'}")
    print(f"搜索测试: {'✓' if results['search_test_success'] else '✗'}")
    
    if results["evaluation_results"]:
        print(f"\n基础评估结果:")
        for metric, value in results["evaluation_results"].items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
    
    if results["retrieval_evaluation_results"]:
        print(f"\n检索评估结果:")
        for metric, value in results["retrieval_evaluation_results"].items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
    
    if results["errors"]:
        print(f"\n错误信息:")
        for error in results["errors"]:
            print(f"  - {error}")


if __name__ == "__main__":
    main()