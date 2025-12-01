# Ragas 评估修复计划

## 问题分析

### 问题描述
运行 Ragas 评估时出现错误：
```
The metric [faithfulness] that is used requires the following additional columns ['response'] to be present in the dataset.
```

### 根本原因
1. **Faithfulness 指标**需要以下字段：
   - `user_input` (问题)
   - `response` (模型生成的回答)
   - `retrieved_contexts` (检索到的上下文)

2. **AnswerRelevancy 指标**需要以下字段：
   - `user_input` (问题)
   - `response` (模型生成的回答)

3. **ContextPrecision 指标**需要以下字段：
   - `user_input` (问题)
   - `retrieved_contexts` (检索到的上下文)
   - `reference` (参考答案)

4. **ContextRecall 指标**需要以下字段：
   - `user_input` (问题)
   - `retrieved_contexts` (检索到的上下文)
   - `reference` (参考答案)

当前的数据集创建逻辑中，`create_retrieval_evaluation_dataset` 方法只创建了 `user_input`、`retrieved_contexts` 和 `reference` 字段，但缺少了 `response` 字段。

## 解决方案

### 1. 修改 `create_retrieval_evaluation_dataset` 方法

在 `ragas_siliconflow_milvus_test.py` 文件中，需要修改 `create_retrieval_evaluation_dataset` 方法，使其能够：

1. 使用 LLM 基于检索到的上下文生成真实的响应
2. 将生成的响应添加到 `SingleTurnSample` 的 `response` 字段中

### 2. 添加响应生成逻辑

需要添加一个新方法 `_generate_response_from_context`，该方法：
- 接收问题和检索到的上下文
- 使用配置的 LLM 生成基于上下文的回答
- 返回生成的回答

### 3. 修改数据集创建流程

在创建每个样本时：
1. 使用 Milvus 检索相关文档
2. 基于检索到的上下文使用 LLM 生成回答
3. 创建包含所有必需字段的 `SingleTurnSample`

## 实现细节

### 修改 `create_retrieval_evaluation_dataset` 方法

```python
def create_retrieval_evaluation_dataset(self) -> Optional[EvaluationDataset]:
    """创建基于检索的评估数据集"""
    try:
        print("\n=== 创建基于检索的评估数据集 ===")
        
        # 获取原始查询样本
        if not self.dataset_extractor:
            raise RuntimeError("数据集提取器未初始化")
        if not self.milvus or not self.embeddings:
            raise RuntimeError("Milvus 或嵌入模型未初始化")
        if not self.evaluator_llm:
            raise RuntimeError("评估 LLM 未初始化，无法生成响应")

        query_samples = self.dataset_extractor.query_samples
        if not query_samples:
            raise ValueError("没有可用的查询样本，请先创建评估数据集")
        
        retrieval_samples = []
        
        for i, query_sample in enumerate(query_samples):
            print(f"处理样本 {i+1}/{len(query_samples)}: {query_sample.question[:30]}...")
            
            # 使用 Milvus 检索相关文档
            retrieved_docs = self.milvus.search(
                query_sample.question,
                self.embeddings,
                top_k=5
            )
            
            # 提取检索到的内容
            retrieved_contexts = [doc["content"] for doc in retrieved_docs]
            
            # 使用 LLM 基于上下文生成回答
            generated_response = self._generate_response_from_context(
                query_sample.question, 
                retrieved_contexts
            )
            
            # 创建 Ragas 样本
            sample = SingleTurnSample(
                user_input=query_sample.question,
                response=generated_response,  # 添加生成的响应
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
```

### 添加响应生成方法

```python
def _generate_response_from_context(self, question: str, contexts: List[str]) -> str:
    """
    基于检索到的上下文使用 LLM 生成回答
    
    Args:
        question: 用户问题
        contexts: 检索到的上下文列表
        
    Returns:
        str: 生成的回答
    """
    try:
        # 构建提示词
        context_text = "\n\n".join([f"上下文 {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        prompt = f"""请基于以下上下文回答问题。请确保你的回答完全基于提供的上下文，不要添加外部信息。

{context_text}

问题: {question}

回答:"""
        
        # 使用 LLM 生成回答
        from openai import AsyncOpenAI
        import asyncio
        
        client = AsyncOpenAI(
            api_key=self.config["llm_api_key"],
            base_url=self.config.get("llm_base_url", "https://api.openai.com/v1")
        )
        
        # 同步调用异步方法
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response = loop.run_until_complete(
                client.chat.completions.create(
                    model=self.config.get("evaluator_model", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "你是一个有帮助的助手，请基于提供的上下文准确回答问题。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
            )
            answer = response.choices[0].message.content.strip()
        finally:
            loop.close()
        
        return answer
        
    except Exception as e:
        print(f"生成回答时出错: {e}")
        # 如果生成失败，返回一个基于上下文的简单回答
        return f"基于提供的上下文，无法完全回答该问题。"
```

### 修改 `run_complete_test` 方法

确保在调用 `create_retrieval_evaluation_dataset` 之前检查 LLM 是否已初始化：

```python
# 7. 创建基于检索的评估数据集
if self.evaluator_llm:
    retrieval_dataset = self.create_retrieval_evaluation_dataset()
    if retrieval_dataset:
        # 8. 运行检索评估
        retrieval_evaluation_results = self.run_ragas_evaluation(retrieval_dataset)
        results["retrieval_evaluation_results"] = retrieval_evaluation_results
else:
    print("⚠ 未配置评估 LLM，跳过检索评估")
```

## 测试计划

1. 修改代码后，运行完整的测试流程
2. 确认所有四个评估指标都能正常运行
3. 检查评估结果是否合理
4. 验证生成的响应是否基于检索到的上下文

## 预期结果

修复后，评估流程应该能够：
1. 成功运行所有四个评估指标
2. 生成包含所有必需字段的评估数据集
3. 提供有意义的评估结果

## 注意事项

1. 确保 LLM API 密钥已正确配置
2. 生成的响应质量取决于 LLM 的性能和提示词设计
3. 如果 LLM 调用失败，需要有后备方案
4. 考虑添加响应生成的错误处理和重试机制