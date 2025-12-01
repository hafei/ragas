"""
SiliconFlow API 自定义嵌入模型
用于与 Ragas 集成的 SiliconFlow 嵌入服务
"""

import requests
import json
import asyncio
import aiohttp
from typing import List, Optional, Dict, Any
from ragas.embeddings.base import BaseRagasEmbedding as BaseEmbedding
import numpy as np


class SiliconFlowEmbeddings(BaseEmbedding):
    """
    SiliconFlow API 嵌入模型实现
    
    Args:
        api_key: SiliconFlow API 密钥
        model_name: 模型名称，默认为 "BAAI/bge-large-zh-v1.5"
        base_url: API 基础 URL，默认为 "https://api.siliconflow.cn/v1"
        batch_size: 批处理大小，默认为 32
    """
    
    PROVIDER_NAME = "siliconflow"
    DEFAULT_MODEL = "BAAI/bge-large-zh-v1.5"
    
    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_MODEL,
        base_url: str = "https://api.siliconflow.cn/v1",
        batch_size: int = 32,
        **kwargs
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.batch_size = batch_size
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def embed_text(self, text: str, **kwargs: Any) -> List[float]:
        """获取单个文本的嵌入向量"""
        return self._get_embeddings_batch([text])[0]
    
    async def aembed_text(self, text: str, **kwargs: Any) -> List[float]:
        """异步获取单个文本的嵌入向量"""
        embeddings = await self._aget_embeddings_batch([text])
        return embeddings[0]
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """获取一批文本的嵌入向量"""
        url = f"{self.base_url}/embeddings"
        
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # 提取嵌入向量
            embeddings = []
            if "data" in result:
                for item in result["data"]:
                    embeddings.append(item["embedding"])
            
            return embeddings
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"SiliconFlow API 请求失败: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"SiliconFlow API 响应解析失败: {e}")
        except KeyError as e:
            raise Exception(f"SiliconFlow API 响应格式错误: {e}")
    
    async def _aget_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """异步获取一批文本的嵌入向量"""
        url = f"{self.base_url}/embeddings"
        
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float"
        }
        
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    # 提取嵌入向量
                    embeddings = []
                    if "data" in result:
                        for item in result["data"]:
                            embeddings.append(item["embedding"])
                    
                    return embeddings
                    
        except aiohttp.ClientError as e:
            raise Exception(f"SiliconFlow API 异步请求失败: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"SiliconFlow API 响应解析失败: {e}")
        except KeyError as e:
            raise Exception(f"SiliconFlow API 响应格式错误: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """获取多个文档的嵌入向量（批处理）"""
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._get_embeddings_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步获取多个文档的嵌入向量（批处理）"""
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = await self._aget_embeddings_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """获取查询文本的嵌入向量"""
        return self.embed_text(text)
    
    async def aembed_query(self, text: str) -> List[float]:
        """异步获取查询文本的嵌入向量"""
        return await self.aembed_text(text)
    
    def get_embedding_dimension(self) -> int:
        """获取嵌入向量的维度"""
        # 对于 BAAI/bge-large-zh-v1.5 模型，维度是 1024
        # 可以通过实际调用来获取维度
        test_embedding = self.embed_text("测试")
        return len(test_embedding)
    
    def similarity(self, query_embedding: List[float], document_embedding: List[float]) -> float:
        """计算两个嵌入向量之间的余弦相似度"""
        query_vec = np.array(query_embedding)
        doc_vec = np.array(document_embedding)
        
        # 计算余弦相似度
        dot_product = np.dot(query_vec, doc_vec)
        norm_query = np.linalg.norm(query_vec)
        norm_doc = np.linalg.norm(doc_vec)
        
        if norm_query == 0 or norm_doc == 0:
            return 0.0
        
        return dot_product / (norm_query * norm_doc)


# 使用示例
if __name__ == "__main__":
    # 需要设置环境变量或直接传入 API 密钥
    import os
    
    api_key = os.getenv("SILICONFLOW_API_KEY", "your-api-key-here")
    
    if api_key == "your-api-key-here":
        print("请设置 SILICONFLOW_API_KEY 环境变量")
    else:
        # 创建嵌入模型实例
        embeddings = SiliconFlowEmbeddings(api_key=api_key)
        
        # 测试嵌入
        test_text = "这是一个测试文本"
        embedding = embeddings.embed_text(test_text)
        
        print(f"嵌入向量维度: {len(embedding)}")
        print(f"前10个维度: {embedding[:10]}")
        
        # 测试批处理
        texts = ["文本1", "文本2", "文本3"]
        batch_embeddings = embeddings.embed_documents(texts)
        print(f"批处理嵌入数量: {len(batch_embeddings)}")
        print(f"每个嵌入维度: {[len(emb) for emb in batch_embeddings]}")