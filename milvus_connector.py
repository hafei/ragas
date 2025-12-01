"""
Milvus 向量数据库连接和操作类
用于与 Ragas 集成的 Milvus 向量数据库操作
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusException
)
from siliconflow_embeddings import SiliconFlowEmbeddings


class MilvusConnector:
    """
    Milvus 向量数据库连接器
    
    Args:
        host: Milvus 服务器地址，默认为 "localhost"
        port: Milvus 服务器端口，默认为 19530
        user: Milvus 用户名（可选）
        password: Milvus 密码（可选）
        db_name: 数据库名称，默认为 "ragas_db"
        collection_name: 集合名称，默认为 "documents"
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        user: Optional[str] = None,
        password: Optional[str] = None,
        db_name: str = "ragas_db",
        collection_name: str = "documents"
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db_name = db_name
        self.collection_name = collection_name
        self.collection = None
        self.connected = False
    
    def connect(self) -> bool:
        """连接到 Milvus 服务器"""
        try:
            # 构建连接参数
            connect_params = {
                "host": self.host,
                "port": self.port
            }
            
            if self.user and self.password:
                connect_params["user"] = self.user
                connect_params["password"] = self.password
            
            # 连接到 Milvus
            connections.connect(**connect_params)
            self.connected = True
            print(f"成功连接到 Milvus: {self.host}:{self.port}")
            return True
            
        except MilvusException as e:
            print(f"连接 Milvus 失败: {e}")
            return False
        except Exception as e:
            print(f"连接错误: {e}")
            return False
    
    def disconnect(self):
        """断开与 Milvus 的连接"""
        if self.connected:
            connections.disconnect("default")
            self.connected = False
            print("已断开与 Milvus 的连接")
    
    def create_collection(
        self,
        dimension: int,
        description: str = "Ragas 评估文档集合"
    ) -> bool:
        """
        创建集合
        
        Args:
            dimension: 向量维度
            description: 集合描述
        
        Returns:
            bool: 创建是否成功
        """
        try:
            # 检查集合是否已存在
            if utility.has_collection(self.collection_name):
                print(f"集合 {self.collection_name} 已存在")
                self.collection = Collection(self.collection_name)
                return True
            
            # 定义字段
            id_field = FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=100,
                is_primary=True,
                description="文档ID"
            )
            
            content_field = FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="文档内容"
            )
            
            metadata_field = FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
                description="文档元数据"
            )
            
            embedding_field = FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=dimension,
                description="文档嵌入向量"
            )
            
            # 创建集合模式
            schema = CollectionSchema(
                fields=[id_field, content_field, metadata_field, embedding_field],
                description=description
            )
            
            # 创建集合
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using='default'
            )
            
            print(f"成功创建集合: {self.collection_name}")
            return True
            
        except MilvusException as e:
            print(f"创建集合失败: {e}")
            return False
        except Exception as e:
            print(f"创建集合错误: {e}")
            return False
    
    def insert_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: SiliconFlowEmbeddings
    ) -> bool:
        """
        插入文档到集合
        
        Args:
            documents: 文档列表，每个文档包含 id, content, metadata
            embeddings: 嵌入模型实例
        
        Returns:
            bool: 插入是否成功
        """
        try:
            if not self.collection:
                print("请先创建或加载集合")
                return False
            
            # 准备数据
            ids = []
            contents = []
            metadatas = []
            embedding_vectors = []
            
            # 批量生成嵌入向量
            texts = [doc["content"] for doc in documents]
            batch_embeddings = embeddings.embed_documents(texts)
            
            for i, doc in enumerate(documents):
                ids.append(doc["id"])
                contents.append(doc["content"])
                metadatas.append(doc.get("metadata", {}))
                embedding_vectors.append(batch_embeddings[i])
            
            # 插入数据
            data = [
                ids,
                contents,
                metadatas,
                embedding_vectors
            ]
            
            insert_result = self.collection.insert(data)
            self.collection.flush()
            
            print(f"成功插入 {len(documents)} 个文档")
            return True
            
        except MilvusException as e:
            print(f"插入文档失败: {e}")
            return False
        except Exception as e:
            print(f"插入文档错误: {e}")
            return False
    
    def create_index(self, index_type: str = "HNSW", metric_type: str = "COSINE") -> bool:
        """
        创建向量索引
        
        Args:
            index_type: 索引类型，默认为 "HNSW"
            metric_type: 距离度量类型，默认为 "COSINE"
        
        Returns:
            bool: 创建是否成功
        """
        try:
            if not self.collection:
                print("请先创建或加载集合")
                return False
            
            # 定义索引参数
            index_params = {
                "metric_type": metric_type,
                "index_type": index_type,
                "params": {
                    "M": 16,
                    "efConstruction": 200
                }
            }
            
            # 创建索引
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            print(f"成功创建 {index_type} 索引")
            return True
            
        except MilvusException as e:
            print(f"创建索引失败: {e}")
            return False
        except Exception as e:
            print(f"创建索引错误: {e}")
            return False
    
    def load_collection(self) -> bool:
        """加载集合到内存"""
        try:
            if not utility.has_collection(self.collection_name):
                print(f"集合 {self.collection_name} 不存在")
                return False
            
            self.collection = Collection(self.collection_name)
            self.collection.load()
            print(f"成功加载集合: {self.collection_name}")
            return True
            
        except MilvusException as e:
            print(f"加载集合失败: {e}")
            return False
        except Exception as e:
            print(f"加载集合错误: {e}")
            return False
    
    def search(
        self,
        query_text: str,
        embeddings: SiliconFlowEmbeddings,
        top_k: int = 5,
        search_params: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似文档
        
        Args:
            query_text: 查询文本
            embeddings: 嵌入模型实例
            top_k: 返回结果数量
            search_params: 搜索参数
        
        Returns:
            List[Dict]: 搜索结果列表
        """
        try:
            if not self.collection:
                print("请先创建或加载集合")
                return []
            
            # 生成查询向量
            query_embedding = embeddings.embed_text(query_text)
            
            # 默认搜索参数
            if search_params is None:
                search_params = {
                    "metric_type": "COSINE",
                    "params": {
                        "ef": 64
                    }
                }
            
            # 执行搜索
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["id", "content", "metadata"]
            )
            
            # 格式化结果
            formatted_results = []
            for hit in results[0]:
                formatted_results.append({
                    "id": hit.entity.get("id"),
                    "content": hit.entity.get("content"),
                    "metadata": hit.entity.get("metadata"),
                    "score": hit.score
                })
            
            return formatted_results
            
        except MilvusException as e:
            print(f"搜索失败: {e}")
            return []
        except Exception as e:
            print(f"搜索错误: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            if not self.collection:
                return {"error": "集合未加载"}
            
            stats = {
                "name": self.collection_name,
                "num_entities": self.collection.num_entities,
                "schema": str(self.collection.schema)
            }
            
            # 获取索引信息
            indexes = self.collection.indexes
            if indexes:
                stats["indexes"] = [str(index) for index in indexes]
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}


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
        
        # 创建 Milvus 连接器
        milvus = MilvusConnector()
        
        # 连接到 Milvus
        if milvus.connect():
            # 创建集合
            dimension = embeddings.get_embedding_dimension()
            if milvus.create_collection(dimension):
                # 创建索引
                milvus.create_index()
                
                # 示例文档
                documents = [
                    {
                        "id": "test1",
                        "content": "这是一个测试文档",
                        "metadata": {"category": "test"}
                    }
                ]
                
                # 插入文档
                milvus.insert_documents(documents, embeddings)
                
                # 搜索测试
                results = milvus.search("测试文档", embeddings)
                print(f"搜索结果: {results}")
                
                # 获取统计信息
                stats = milvus.get_collection_stats()
                print(f"集合统计: {stats}")
        
        # 断开连接
        milvus.disconnect()