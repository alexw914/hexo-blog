---
title: Transformer结构及其变体
date: 2026-11-26 16:30:00
tags:
  - Transformer
  - LLM
  - 注意力机制
  - 模型结构
toc:
  number: false
  max_depth: 3
---

本文梳理标准 Transformer 的核心组件，以及面向大模型训练与推理效率演进出的主要结构变体（如 Pre-Norm、RMSNorm、RoPE、GQA/MLA、SwiGLU 与 MoE 等）。

> 注：本文为结构大纲，具体细节与公式推导后续持续补充。

---

## 一、标准 Transformer 架构回顾

### 1.1 Encoder-Decoder 整体结构
- 输入 Embedding 与位置编码
- 多层残差连接与归一化（Post-Norm vs Pre-Norm）
- 因果掩码（Causal Mask）与自回归生成

### 1.2 Scaled Dot-Product Attention 与 MHA
- Q / K / V 投影与缩放点积计算
- 多头注意力（Multi-Head Attention）的表达能力与计算复杂度（时间 `O(N^2 d)`、空间 `O(N^2)`）

### 1.3 Feed-Forward Network（FFN）
- 两层全连接 + 激活函数结构
- 参数量占比与计算访存特征

---

## 二、归一化与激活函数演进

### 2.1 Post-Norm 到 Pre-Norm
- 深层网络梯度稳定性对比
- 现代 LLM（LLaMA / Qwen 等）的默认选择

### 2.2 LayerNorm 到 RMSNorm
- 去除均值中心化带来的计算与访存收益
- 数值稳定性与算子融合友好性

### 2.3 ReLU / GELU 到 SwiGLU
- 门控线性单元（GLU）结构与维度配比（如 `8/3 d`）
- 表达能力提升与显存开销权衡

---

## 三、位置编码演进

### 3.1 绝对位置编码（Sinusoidal / Learnable）
- 原理与长度外推局限

### 3.2 相对位置编码与 ALiBi
- Attention Bias 思路与外推能力

### 3.3 旋转位置编码（RoPE）及长度扩展
- 复数旋转与相对位置内积性质
- NTK-Aware、YaRN 等长上下文扩展方法

---

## 四、注意力机制的推理友好变体

### 4.1 KV Cache 与访存瓶颈
- Prefill（计算密集）与 Decode（访存密集）差异
- KV Cache 显存占用随上下文长度与 Batch Size 的增长

### 4.2 MHA → MQA → GQA
- 多查询注意力（MQA）与分组查询注意力（GQA）对 KV Cache 显存与带宽的压缩
- 精度与推理吞吐的折中

### 4.3 Multi-head Latent Attention（MLA）
- 低秩键值联合压缩与解耦 RoPE 思路（DeepSeek 系列）
- 权重吸收（Weight Absorption）对推理阶段的作用

### 4.4 稀疏注意力与线性注意力
- Sliding Window Attention（Mistral 等）
- 线性注意力 / 状态空间模型（Mamba 等）混合架构简介

---

## 五、稀疏扩展：Mixture of Experts（MoE）

### 5.1 门控路由（Router）与 Top-K 专家选择
- 稀疏激活如何解耦参数规模与单 Token 计算量
- 负载均衡损失（Auxiliary Loss / 无辅助损失策略）

### 5.2 Shared Expert 与 Fine-Grained Expert
- 共享专家与细粒度专家切分设计

### 5.3 MoE 的部署与通信挑战
- Expert Parallelism（EP）与 All-to-All 通信开销
- 显存占用与专家负载均衡

---

## 六、主流架构对照总结

| 模型系列 | 架构类型 | 归一化 | 位置编码 | 注意力变体 | FFN / 稀疏结构 |
|---|---|---|---|---|---|
| 原始 Transformer | Encoder-Decoder | Post-LN | Sinusoidal | MHA | ReLU FFN |
| GPT-2 / GPT-3 | Decoder-Only | Pre-LN | Learnable | MHA | GELU FFN |
| LLaMA 2 / 3 | Decoder-Only | RMSNorm | RoPE | GQA | SwiGLU |
| Qwen 2.5 / 3 | Decoder-Only | RMSNorm | RoPE | GQA | SwiGLU / MoE |
| DeepSeek-V2 / V3 | Decoder-Only | RMSNorm | Decoupled RoPE | MLA | Fine-Grained MoE |
