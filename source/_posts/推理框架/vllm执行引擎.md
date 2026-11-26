---
title: vllm执行引擎
date: 2026-11-26 16:35:00
tags:
  - vLLM
  - 推理框架
  - PagedAttention
  - LLM部署
toc:
  number: false
  max_depth: 3
---

本文梳理 vLLM 推理框架的执行引擎架构，重点拆解其调度机制、PagedAttention 与 KV Cache 显存管理、Prefill/Decode 执行链路以及核心性能优化手段。

> 注：本文为结构大纲，后续将结合源码（含 vLLM V1 架构演进）逐步补充调用链路与性能实验。

---

## 一、vLLM 整体架构概览

### 1.1 请求生命周期
- API Server（OpenAI 兼容接口）→ Tokenizer → EngineCore / LLMEngine → Scheduler → Worker / ModelRunner → Detokenizer
- 异步流式返回与进程间通信（ZMQ / 多进程解耦）

### 1.2 核心模块职责划分
- **Scheduler**：请求队列管理（Waiting / Running / Swapped）与每步调度决策
- **KV Cache Manager（BlockManager）**：物理块分配、前缀复用与回收
- **Executor & Worker**：单卡 / 多卡（TP / PP）分布式执行协调
- **ModelRunner**：输入打包、Attention Metadata 构建、模型前向与采样

---

## 二、PagedAttention 与 KV Cache 管理

### 2.1 传统连续显存分配的痛点
- 内部碎片（按 `max_model_len` 预分配浪费）与外部碎片
- 显存瓶颈如何限制最大并发 Batch Size

### 2.2 逻辑块与物理块映射
- Block Table 设计思路（类比操作系统虚拟内存分页）
- Copy-on-Write 机制与多采样（Beam Search / Parallel Sampling）共享

### 2.3 Prefix Caching（自动前缀缓存）
- 基于 Block Hash 的 KV 块复用
- 适用场景（System Prompt、多轮对话、RAG 共享上下文）

---

## 三、调度器与 Continuous Batching

### 3.1 Iteration-Level Scheduling（迭代级调度）
- Static Batching 与 Continuous Batching 的吞吐对比
- 请求动态加入与提前结束释放资源

### 3.2 Chunked Prefill（分块预填充）
- 长 Prompt Prefill 阻塞 Decode 导致 ITL/TPOT 抖动的问题
- Prefill 与 Decode 混合组批（Token Budget 控制）

### 3.3 抢占（Preemption）与显存压力处理
- Recompute（重计算）与 Swap（换出至 CPU 内存）策略对比

---

## 四、模型执行链路与底层优化

### 4.1 ModelRunner 执行流程
- `prepare_input`：构建 token ids、positions、slot mapping 与 block tables
- Forward Pass：Attention Backend（FlashAttention / FlashInfer / PagedAttention Kernel）调度
- Sampler：Temperature / Top-P / Top-K / Repetition Penalty 与采样加速

### 4.2 CUDA Graph 与 Torch Compile
- Decode 阶段小 Batch 下的 Python / Kernel Launch 开销
- 分档捕获（Capture Batch Sizes）与图复用限制

### 4.3 高级推理特性
- **Speculative Decoding（投机解码）**：Draft Model / EAGLE / MTP 验证流程
- **量化推理支持**：FP8、AWQ、GPTQ、W8A8 INT8 与 Marlin / Cutlass Kernel
- **分布式并行**：Tensor Parallelism（TP）、Pipeline Parallelism（PP）、Data Parallelism（DP）与 EP（MoE）

---

## 五、vLLM V0 到 V1 架构演进要点

- 多进程架构重构：API Server 与 EngineCore 彻底解耦，降低 CPU 调度开销
- 统一调度器设计：将 Prefill 与 Decode 统一为基于 Token 预算的调度视图
- 持久化 Batch 状态与零开销（Zero-Overhead）Prefix Caching 优化

---

## 六、关键性能指标与调优 checklist

- 核心指标：TTFT（首字延迟）、TPOT / ITL（单 Token 延迟）、Throughput（Tokens/s）
- 常用参数调优：`gpu_memory_utilization`、`max_num_seqs`、`max_num_batched_tokens`、`enable_prefix_caching`、`enable_chunked_prefill`
