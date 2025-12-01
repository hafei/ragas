"""
测试异步修复的简单脚本
验证事件循环和客户端资源管理是否正常工作
"""

import asyncio
import os
import json
from siliconflow_embeddings import SiliconFlowEmbeddings
from ragas_siliconflow_milvus_test import RagasSiliconFlowMilvusTest, load_config


async def test_siliconflow_embeddings():
    """测试 SiliconFlow 嵌入模型的异步功能"""
    print("=== 测试 SiliconFlow 嵌入模型 ===")
    
    # 获取 API 密钥
    api_key = os.getenv("SILICONFLOW_API_KEY", "")
    if not api_key:
        print("警告: 未设置 SILICONFLOW_API_KEY，跳过嵌入模型测试")
        return True
    
    try:
        # 创建嵌入模型实例
        embeddings = SiliconFlowEmbeddings(api_key=api_key)
        
        # 测试同步嵌入
        print("1. 测试同步嵌入...")
        test_text = "这是一个测试文本"
        embedding = embeddings.embed_text(test_text)
        print(f"成功: 同步嵌入成功，维度: {len(embedding)}")
        
        # 测试异步嵌入
        print("2. 测试异步嵌入...")
        async_embedding = await embeddings.aembed_text(test_text)
        print(f"成功: 异步嵌入成功，维度: {len(async_embedding)}")
        
        # 测试批处理异步嵌入
        print("3. 测试批处理异步嵌入...")
        test_texts = ["文本1", "文本2", "文本3"]
        batch_embeddings = await embeddings.aembed_documents(test_texts)
        print(f"成功: 批处理异步嵌入成功，数量: {len(batch_embeddings)}")
        
        return True
        
    except Exception as e:
        print(f"失败: SiliconFlow 嵌入模型测试失败: {e}")
        return False


def test_ragas_integration():
    """测试 Ragas 集成"""
    print("\n=== 测试 Ragas 集成 ===")
    
    try:
        # 加载配置
        config = load_config()
        
        # 创建测试实例
        test = RagasSiliconFlowMilvusTest(config)
        
        # 测试组件初始化
        print("1. 测试组件初始化...")
        if not test.setup():
            print("失败: 组件初始化失败")
            return False
        print("成功: 组件初始化成功")
        
        # 测试嵌入维度获取
        print("2. 测试嵌入维度获取...")
        dimension = test.embeddings.get_embedding_dimension()
        print(f"成功: 嵌入维度: {dimension}")
        
        # 测试响应生成（如果有 LLM 配置）
        if test.evaluator_llm:
            print("3. 测试响应生成...")
            test_contexts = ["这是测试上下文1", "这是测试上下文2"]
            response = test._generate_response_from_context("测试问题", test_contexts)
            print(f"✓ 响应生成成功: {response[:50]}...")
        else:
            print("警告: 未配置 LLM，跳过响应生成测试")
        
        # 清理资源
        if test.milvus:
            test.milvus.disconnect()
        
        return True
        
    except Exception as e:
        print(f"失败: Ragas 集成测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("异步修复测试")
    print("=" * 50)
    
    results = []
    
    # 测试 SiliconFlow 嵌入模型
    result1 = await test_siliconflow_embeddings()
    results.append(("SiliconFlow 嵌入模型", result1))
    
    # 测试 Ragas 集成
    result2 = test_ragas_integration()
    results.append(("Ragas 集成", result2))
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print("测试结果摘要:")
    for test_name, success in results:
        status = "成功" if success else "失败"
        print(f"{test_name}: {status}")
    
    all_success = all(success for _, success in results)
    if all_success:
        print("\n恭喜: 所有测试通过！异步修复成功。")
    else:
        print("\n错误: 部分测试失败，需要进一步检查。")
    
    return all_success


if __name__ == "__main__":
    # 运行测试
    success = asyncio.run(main())
    exit(0 if success else 1)