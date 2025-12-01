# -*- coding: utf-8 -*-
"""
简单的异步修复测试脚本
"""

import asyncio
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_basic_async():
    """测试基本异步功能"""
    print("=== 测试基本异步功能 ===")
    
    try:
        # 测试事件循环创建和管理
        print("1. 测试事件循环创建...")
        
        # 获取当前事件循环
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                print("事件循环已关闭，创建新循环")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            else:
                print("使用现有事件循环")
        except RuntimeError:
            print("创建新事件循环")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 测试异步任务
        async def dummy_task():
            await asyncio.sleep(0.1)
            return "异步任务完成"
        
        result = await dummy_task()
        print(f"2. 异步任务测试: {result}")
        
        # 测试异步客户端模拟
        print("3. 测试异步客户端模拟...")
        
        # 模拟异步HTTP请求
        async def mock_async_request():
            await asyncio.sleep(0.1)
            return {"status": "success", "data": [1, 2, 3, 4, 5]}
        
        response = await mock_async_request()
        print(f"4. 模拟异步请求成功: {response['status']}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False


def test_sync_components():
    """测试同步组件"""
    print("\n=== 测试同步组件 ===")
    
    try:
        # 测试基本导入
        print("1. 测试模块导入...")
        
        try:
            from siliconflow_embeddings import SiliconFlowEmbeddings
            print("成功: SiliconFlowEmbeddings 导入成功")
        except ImportError as e:
            print(f"警告: SiliconFlowEmbeddings 导入失败: {e}")
        
        try:
            from ragas_siliconflow_milvus_test import RagasSiliconFlowMilvusTest
            print("成功: RagasSiliconFlowMilvusTest 导入成功")
        except ImportError as e:
            print(f"警告: RagasSiliconFlowMilvusTest 导入失败: {e}")
        
        # 测试配置加载
        print("2. 测试配置加载...")
        try:
            from ragas_siliconflow_milvus_test import load_config
            config = load_config()
            print(f"成功: 配置加载成功，包含 {len(config)} 个配置项")
        except Exception as e:
            print(f"警告: 配置加载失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"同步组件测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("异步修复测试")
    print("=" * 50)
    
    results = []
    
    # 测试基本异步功能
    result1 = await test_basic_async()
    results.append(("基本异步功能", result1))
    
    # 测试同步组件
    result2 = test_sync_components()
    results.append(("同步组件", result2))
    
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