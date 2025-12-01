"""
从 JSON 文档提取数据集的功能
用于将 JSON 文档转换为 Ragas 可用的数据集格式
"""

import json
import os
import random
from typing import List, Dict, Any, Optional, Tuple
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from dataclasses import dataclass


@dataclass
class Document:
    """文档数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any]


@dataclass
class QuerySample:
    """查询样本数据结构"""
    question: str
    expected_answer: str
    context: List[str]
    metadata: Optional[Dict[str, Any]] = None


class JSONDatasetExtractor:
    """
    JSON 数据集提取器
    
    Args:
        json_file_path: JSON 文件路径
        encoding: 文件编码，默认为 "utf-8"
    """
    
    def __init__(self, json_file_path: str, encoding: str = "utf-8"):
        self.json_file_path = json_file_path
        self.encoding = encoding
        self.documents = []
        self.query_samples = []
    
    def load_documents(self) -> List[Document]:
        """
        从 JSON 文件加载文档
        
        Returns:
            List[Document]: 文档列表
        
        Expected JSON format:
        [
            {
                "id": "doc1",
                "content": "文档内容",
                "metadata": {"category": "concept", "difficulty": "easy"}
            },
            ...
        ]
        """
        try:
            if not os.path.exists(self.json_file_path):
                raise FileNotFoundError(f"文件不存在: {self.json_file_path}")
            
            with open(self.json_file_path, 'r', encoding=self.encoding) as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("JSON 文件应该包含一个文档数组")
            
            documents = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                # 验证必需字段
                if "id" not in item or "content" not in item:
                    print(f"跳过缺少必需字段的文档: {item}")
                    continue
                
                doc = Document(
                    id=item["id"],
                    content=item["content"],
                    metadata=item.get("metadata", {})
                )
                documents.append(doc)
            
            self.documents = documents
            print(f"成功加载 {len(documents)} 个文档")
            return documents
            
        except json.JSONDecodeError as e:
            raise Exception(f"JSON 解析错误: {e}")
        except Exception as e:
            raise Exception(f"加载文档失败: {e}")
    
    def generate_query_samples(
        self,
        num_samples: int = 10,
        min_context_length: int = 2,
        max_context_length: int = 5
    ) -> List[QuerySample]:
        """
        生成查询样本
        
        Args:
            num_samples: 生成的样本数量
            min_context_length: 最小上下文长度
            max_context_length: 最大上下文长度
        
        Returns:
            List[QuerySample]: 查询样本列表
        """
        if not self.documents:
            raise ValueError("请先加载文档")
        
        if len(self.documents) < min_context_length:
            raise ValueError(f"文档数量不足，至少需要 {min_context_length} 个文档")
        
        query_samples = []
        
        for i in range(num_samples):
            # 随机选择上下文长度
            context_length = random.randint(min_context_length, max_context_length)
            
            # 随机选择文档作为上下文
            context_docs = random.sample(self.documents, context_length)
            context_texts = [doc.content for doc in context_docs]
            
            # 生成问题和期望答案
            question, expected_answer = self._generate_qa_from_context(context_docs)
            
            # 创建查询样本
            sample = QuerySample(
                question=question,
                expected_answer=expected_answer,
                context=context_texts,
                metadata={
                    "context_doc_ids": [doc.id for doc in context_docs],
                    "context_categories": [doc.metadata.get("category", "unknown") for doc in context_docs]
                }
            )
            
            query_samples.append(sample)
        
        self.query_samples = query_samples
        print(f"成功生成 {len(query_samples)} 个查询样本")
        return query_samples
    
    def _generate_qa_from_context(self, context_docs: List[Document]) -> Tuple[str, str]:
        """
        从上下文文档生成问题和答案
        
        Args:
            context_docs: 上下文文档列表
        
        Returns:
            Tuple[str, str]: (问题, 期望答案)
        """
        import random
        
        # 简单的 QA 生成策略
        strategies = [
            self._generate_concept_question,
            self._generate_technical_question,
            self._generate_comparison_question
        ]
        
        # 随机选择策略
        strategy = random.choice(strategies)
        return strategy(context_docs)
    
    def _generate_concept_question(self, context_docs: List[Document]) -> Tuple[str, str]:
        """生成概念性问题"""
        # 选择一个包含概念类别的文档
        concept_docs = [doc for doc in context_docs if doc.metadata.get("category") == "concept"]
        
        if concept_docs:
            doc = random.choice(concept_docs)
            question = f"什么是{doc.content[:20]}？"
            expected_answer = doc.content
        else:
            doc = random.choice(context_docs)
            question = f"请解释：{doc.content[:30]}..."
            expected_answer = doc.content
        
        return question, expected_answer
    
    def _generate_technical_question(self, context_docs: List[Document]) -> Tuple[str, str]:
        """生成技术性问题"""
        tech_docs = [doc for doc in context_docs if doc.metadata.get("category") == "technology"]
        
        if tech_docs:
            doc = random.choice(tech_docs)
            question = f"{doc.content[:30]}是如何工作的？"
            expected_answer = doc.content
        else:
            doc = random.choice(context_docs)
            question = f"请详细说明：{doc.content[:30]}..."
            expected_answer = doc.content
        
        return question, expected_answer
    
    def _generate_comparison_question(self, context_docs: List[Document]) -> Tuple[str, str]:
        """生成比较性问题"""
        if len(context_docs) >= 2:
            doc1, doc2 = random.sample(context_docs, 2)
            question = f"比较{doc1.content[:20]}和{doc2.content[:20]}的特点。"
            expected_answer = f"{doc1.content} {doc2.content}"
        else:
            doc = context_docs[0]
            question = f"请分析：{doc.content[:30]}..."
            expected_answer = doc.content
        
        return question, expected_answer
    
    def create_ragas_dataset(self) -> EvaluationDataset:
        """
        创建 Ragas 评估数据集
        
        Returns:
            EvaluationDataset: Ragas 评估数据集
        """
        if not self.query_samples:
            raise ValueError("请先生成查询样本")
        
        samples = []
        
        for query_sample in self.query_samples:
            # 创建 Ragas SingleTurnSample
            sample = SingleTurnSample(
                user_input=query_sample.question,
                retrieved_contexts=query_sample.context,
                reference=query_sample.expected_answer,
                metadata=query_sample.metadata
            )
            samples.append(sample)
        
        # 创建评估数据集
        dataset = EvaluationDataset(samples=samples)
        print(f"成功创建包含 {len(samples)} 个样本的 Ragas 数据集")
        
        return dataset
    
    def save_dataset_to_json(self, output_path: str):
        """
        保存数据集到 JSON 文件
        
        Args:
            output_path: 输出文件路径
        """
        try:
            dataset_data = {
                "documents": [
                    {
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": doc.metadata
                    }
                    for doc in self.documents
                ],
                "query_samples": [
                    {
                        "question": sample.question,
                        "expected_answer": sample.expected_answer,
                        "context": sample.context,
                        "metadata": sample.metadata
                    }
                    for sample in self.query_samples
                ]
            }
            
            with open(output_path, 'w', encoding=self.encoding) as f:
                json.dump(dataset_data, f, ensure_ascii=False, indent=2)
            
            print(f"数据集已保存到: {output_path}")
            
        except Exception as e:
            raise Exception(f"保存数据集失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        if not self.documents or not self.query_samples:
            return {"error": "请先加载文档和生成查询样本"}
        
        stats = {
            "documents": {
                "total_count": len(self.documents),
                "categories": {},
                "avg_content_length": 0
            },
            "query_samples": {
                "total_count": len(self.query_samples),
                "avg_context_length": 0,
                "avg_question_length": 0
            }
        }
        
        # 文档统计
        total_content_length = 0
        for doc in self.documents:
            category = doc.metadata.get("category", "unknown")
            stats["documents"]["categories"][category] = stats["documents"]["categories"].get(category, 0) + 1
            total_content_length += len(doc.content)
        
        stats["documents"]["avg_content_length"] = total_content_length / len(self.documents)
        
        # 查询样本统计
        total_context_length = 0
        total_question_length = 0
        
        for sample in self.query_samples:
            total_context_length += len(sample.context)
            total_question_length += len(sample.question)
        
        stats["query_samples"]["avg_context_length"] = total_context_length / len(self.query_samples)
        stats["query_samples"]["avg_question_length"] = total_question_length / len(self.query_samples)
        
        return stats


# 使用示例
if __name__ == "__main__":
    # 创建数据集提取器
    extractor = JSONDatasetExtractor("test_data.json")
    
    try:
        # 加载文档
        documents = extractor.load_documents()
        print(f"加载了 {len(documents)} 个文档")
        
        # 生成查询样本
        query_samples = extractor.generate_query_samples(num_samples=5)
        print(f"生成了 {len(query_samples)} 个查询样本")
        
        # 创建 Ragas 数据集
        ragas_dataset = extractor.create_ragas_dataset()
        print(f"创建了包含 {len(ragas_dataset.samples)} 个样本的 Ragas 数据集")
        
        # 保存数据集
        extractor.save_dataset_to_json("extracted_dataset.json")
        
        # 获取统计信息
        stats = extractor.get_statistics()
        print("数据集统计信息:")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"错误: {e}")