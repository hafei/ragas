"""
简化的测试脚本
验证 SiliconFlow 嵌入和 JSON 数据集提取功能
"""

import os
import json
from siliconflow_embeddings import SiliconFlowEmbeddings
from json_dataset_extractor import JSONDatasetExtractor


def test_siliconflow_embeddings():
    """测试 SiliconFlow 嵌入功能"""
    print("=== 测试 SiliconFlow 嵌入 ===")
    
    api_key = os.getenv("SILICONFLOW_API_KEY", "your-api-key-here")
    
    if api_key == "your-api-key-here":
        print("⚠ 请设置 SILICONFLOW_API_KEY 环境变量")
        return False
    
    try:
        # 创建嵌入模型
        embeddings = SiliconFlowEmbeddings(api_key=api_key)
        
        # 测试单个文本嵌入
        test_text = "这是一个测试文本"
        embedding = embeddings.embed_text(test_text)
        print(f"✓ 单个文本嵌入成功，维度: {len(embedding)}")
        
        # 测试批量嵌入
        texts = ["文本1", "文本2", "文本3"]
        batch_embeddings = embeddings.embed_documents(texts)
        print(f"✓ 批量嵌入成功，数量: {len(batch_embeddings)}")
        
        # 测试相似度计算
        similarity = embeddings.similarity(embedding, batch_embeddings[0])
        print(f"✓ 相似度计算成功: {similarity:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ SiliconFlow 嵌入测试失败: {e}")
        return False


def test_json_dataset_extractor():
    """测试 JSON 数据集提取功能"""
    print("\n=== 测试 JSON 数据集提取 ===")
    
    try:
        # 创建提取器
        extractor = JSONDatasetExtractor("test_data.json")
        
        # 加载文档
        documents = extractor.load_documents()
        print(f"✓ 加载了 {len(documents)} 个文档")
        
        # 生成查询样本
        query_samples = extractor.generate_query_samples(num_samples=3)
        print(f"✓ 生成了 {len(query_samples)} 个查询样本")
        
        # 显示统计信息
        stats = extractor.get_statistics()
        print("✓ 数据集统计信息:")
        print(f"  文档总数: {stats['documents']['total_count']}")
        print(f"  平均内容长度: {stats['documents']['avg_content_length']:.1f}")
        print(f"  查询样本总数: {stats['query_samples']['total_count']}")
        
        # 保存数据集
        extractor.save_dataset_to_json("extracted_dataset.json")
        print("✓ 数据集已保存到 extracted_dataset.json")
        
        return True
        
    except Exception as e:
        print(f"✗ JSON 数据集提取测试失败: {e}")
        return False


def test_basic_integration():
    """测试基本集成功能"""
    print("\n=== 测试基本集成 ===")
    
    api_key = os.getenv("SILICONFLOW_API_KEY", "your-api-key-here")
    
    if api_key == "your-api-key-here":
        print("⚠ 请设置 SILICONFLOW_API_KEY 环境变量")
        return False
    
    try:
        # 创建嵌入模型
        embeddings = SiliconFlowEmbeddings(api_key=api_key)
        
        # 创建数据集提取器
        extractor = JSONDatasetExtractor("test_data.json")
        
        # 加载文档
        documents = extractor.load_documents()
        
        # 为文档生成嵌入
        texts = [doc.content for doc in documents]
        doc_embeddings = embeddings.embed_documents(texts)
        
        print(f"✓ 为 {len(documents)} 个文档生成了嵌入向量")
        
        # 生成查询样本
        query_samples = extractor.generate_query_samples(num_samples=2)
        
        # 为查询生成嵌入
        queries = [sample.question for sample in query_samples]
        query_embeddings = embeddings.embed_documents(queries)
        
        print(f"✓ 为 {len(queries)} 个查询生成了嵌入向量")
        
        # 计算查询与文档的相似度
        for i, query_emb in enumerate(query_embeddings):
            similarities = []
            for doc_emb in doc_embeddings:
                sim = embeddings.similarity(query_emb, doc_emb)
                similarities.append(sim)
            
            print(f"查询 '{queries[i][:20]}...' 与最相似文档的相似度: {max(similarities):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本集成测试失败: {e}")
        return False


def main():
    """主函数"""
    print("Ragas + SiliconFlow + Milvus 简化测试")
    print("=" * 50)
    
    # 运行测试
    tests = [
        ("SiliconFlow 嵌入", test_siliconflow_embeddings),
        ("JSON 数据集提取", test_json_dataset_extractor),
        ("基本集成", test_basic_integration)
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
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！")
    else:
        print("⚠ 部分测试失败，请检查错误信息")


if __name__ == "__main__":
    main()