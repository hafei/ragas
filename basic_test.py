"""
基础测试脚本
测试 JSON 数据加载和基本功能（不依赖 Ragas）
"""

import json
import os
from typing import List, Dict, Any


def test_json_loading():
    """测试 JSON 数据加载"""
    print("=== 测试 JSON 数据加载 ===")
    
    try:
        # 读取 JSON 文件
        with open("test_data.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ 成功加载 JSON 文件")
        print(f"✓ 包含 {len(data)} 个文档")
        
        # 验证数据格式
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                print(f"✗ 第 {i+1} 项不是字典")
                return False
            
            required_fields = ["id", "content"]
            for field in required_fields:
                if field not in item:
                    print(f"✗ 第 {i+1} 项缺少必需字段: {field}")
                    return False
            
            print(f"  文档 {i+1}: ID={item['id']}, 内容长度={len(item['content'])}")
        
        print("✓ 所有文档格式验证通过")
        return True
        
    except Exception as e:
        print(f"✗ JSON 加载失败: {e}")
        return False


def test_data_structure():
    """测试数据结构和统计"""
    print("\n=== 测试数据结构 ===")
    
    try:
        with open("test_data.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 统计信息
        total_docs = len(data)
        total_content_length = sum(len(doc["content"]) for doc in data)
        avg_content_length = total_content_length / total_docs
        
        categories = {}
        for doc in data:
            category = doc.get("metadata", {}).get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1
        
        print(f"✓ 文档总数: {total_docs}")
        print(f"✓ 平均内容长度: {avg_content_length:.1f}")
        print(f"✓ 类别分布:")
        for category, count in categories.items():
            print(f"    {category}: {count}")
        
        # 生成示例查询
        print("\n✓ 示例查询生成:")
        sample_queries = [
            "什么是向量数据库？",
            "Milvus 有什么特点？",
            "HNSW 算法如何工作？"
        ]
        
        for query in sample_queries:
            # 简单的关键词匹配（模拟检索）
            matching_docs = []
            for doc in data:
                content = doc["content"].lower()
                query_lower = query.lower()
                
                # 简单的关键词匹配
                if any(word in content for word in query_lower.split() if len(word) > 1):
                    matching_docs.append(doc["id"])
            
            print(f"  查询: '{query}' -> 匹配文档: {matching_docs}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据结构测试失败: {e}")
        return False


def test_config_loading():
    """测试配置文件加载"""
    print("\n=== 测试配置文件加载 ===")
    
    try:
        # 读取配置文件
        with open("config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print("✓ 成功加载配置文件")
        
        # 验证必需的配置项
        required_keys = [
            "siliconflow_api_key",
            "llm_provider",
            "llm_base_url",
            "llm_api_key",
            "json_data_path",
            "embedding_model",
            "evaluator_model",
            "milvus_host",
            "milvus_port"
        ]
        
        for key in required_keys:
            if key not in config:
                print(f"✗ 缺少配置项: {key}")
                return False
            if 'key' in key.lower() or 'password' in key.lower():
                print(f"  {key}: ***")
            else:
                print(f"  {key}: {config[key]}")
        
        print("✓ 所有必需配置项都存在")
        return True
        
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False


def test_file_structure():
    """测试文件结构"""
    print("\n=== 测试文件结构 ===")
    
    required_files = [
        "test_data.json",
        "config.json",
        "siliconflow_embeddings.py",
        "milvus_connector.py",
        "json_dataset_extractor.py",
        "simple_test.py",
        "ragas_siliconflow_milvus_test.py",
        "README.md"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"✓ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"✗ {file_path}")
    
    print(f"\n✓ 存在的文件: {len(existing_files)}/{len(required_files)}")
    
    if missing_files:
        print(f"✗ 缺失的文件: {missing_files}")
        return False
    
    return True


def main():
    """主函数"""
    print("Ragas + SiliconFlow + Milvus 基础测试")
    print("=" * 50)
    
    # 运行测试
    tests = [
        ("文件结构", test_file_structure),
        ("JSON 数据加载", test_json_loading),
        ("数据结构", test_data_structure),
        ("配置加载", test_config_loading)
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
        print("🎉 所有基础测试都通过了！")
        print("\n下一步:")
        print("1. 设置 SILICONFLOW_API_KEY 环境变量")
        print("2. 安装完整依赖: pip install ragas pymilvus openai")
        print("3. 运行完整测试: python ragas_siliconflow_milvus_test.py")
    else:
        print("⚠ 部分测试失败，请检查错误信息")


if __name__ == "__main__":
    main()