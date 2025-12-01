"""
测试数据集结构 - 验证 response 字段是否存在
"""

import os
from json_dataset_extractor import JSONDatasetExtractor

def test_dataset_structure():
    """测试数据集结构是否包含 response 字段"""
    print("=== 测试数据集结构 ===")
    
    try:
        # 创建数据集提取器
        extractor = JSONDatasetExtractor("test_data.json")
        
        # 加载文档
        documents = extractor.load_documents()
        print(f"加载了 {len(documents)} 个文档")
        
        # 生成查询样本
        query_samples = extractor.generate_query_samples(num_samples=3)
        print(f"生成了 {len(query_samples)} 个查询样本")
        
        # 创建 Ragas 数据集
        ragas_dataset = extractor.create_ragas_dataset()
        print(f"创建了包含 {len(ragas_dataset.samples)} 个样本的 Ragas 数据集")
        
        # 检查每个样本的字段
        all_samples_valid = True
        for i, sample in enumerate(ragas_dataset.samples):
            print(f"\n样本 {i+1} 的字段:")
            print(f"  user_input: {sample.user_input[:50]}...")
            
            # 检查 response 字段
            if hasattr(sample, 'response') and sample.response:
                print(f"  response: {sample.response[:50]}...")
                print("  [OK] response 字段存在且不为空")
            else:
                print("  [ERROR] response 字段缺失或为空")
                all_samples_valid = False
            
            # 检查其他字段
            context_count = len(sample.retrieved_contexts) if sample.retrieved_contexts else 0
            print(f"  retrieved_contexts: {context_count} 个上下文")
            print(f"  reference: {sample.reference[:50] if sample.reference else 'None'}...")
        
        return all_samples_valid
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False

def test_original_error():
    """测试原始错误是否已解决"""
    print("\n=== 测试原始错误是否已解决 ===")
    
    try:
        # 创建数据集提取器
        extractor = JSONDatasetExtractor("test_data.json")
        
        # 加载文档和生成查询样本
        extractor.load_documents()
        extractor.generate_query_samples(num_samples=2)
        
        # 创建数据集
        dataset = extractor.create_ragas_dataset()
        
        # 尝试使用 faithfulness 指标（这是导致原始错误的原因）
        from ragas.metrics import Faithfulness
        
        # 创建一个简单的 faithfulness 指标实例
        faithfulness_metric = Faithfulness()
        
        # 检查数据集是否包含所需字段
        required_columns = faithfulness_metric.required_columns
        print(f"Faithfulness 指标需要的字段: {required_columns}")
        
        # 检查数据集是否包含这些字段
        sample = dataset.samples[0]
        missing_fields = []
        
        for field in required_columns:
            if not hasattr(sample, field) or getattr(sample, field) is None:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"[ERROR] 缺少字段: {missing_fields}")
            return False
        else:
            print("[OK] 所有必需字段都存在")
            return True
            
    except Exception as e:
        print(f"测试失败: {e}")
        return False

def main():
    """主函数"""
    print("数据集结构测试 - 验证 response 字段修复")
    print("=" * 50)
    
    # 测试数据集结构
    structure_test_passed = test_dataset_structure()
    
    # 测试原始错误是否已解决
    error_test_passed = test_original_error()
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print("测试结果摘要:")
    print(f"数据集结构测试: {'通过' if structure_test_passed else '失败'}")
    print(f"原始错误解决测试: {'通过' if error_test_passed else '失败'}")
    
    if structure_test_passed and error_test_passed:
        print("\n[SUCCESS] 所有测试通过！response 字段问题已解决。")
        print("现在您可以运行 Ragas 评估，不会再出现缺少 response 字段的错误。")
    else:
        print("\n[FAILED] 部分测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()