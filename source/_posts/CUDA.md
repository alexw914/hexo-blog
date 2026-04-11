---
title: CUDA
date: 2026-03-18 22:41:26
categories: 模型部署
tags:
  - CUDA
---

这篇文章整理 CUDA 中常见的执行模型、Warp 级原语、性能分析方法以及典型算子优化方向，适合作为后续继续补充实现细节的提纲。

## 1. CUDA 基础概念

先把几个常见概念区分清楚：

- `grid`
- `block`
- `warp`
- `context`
- `stream`

这些概念分别对应不同层次的执行与调度单位，理解它们是分析 CUDA 程序行为的前提。

## 2. Warp 级原语与规约

Warp 级操作通常用于减少共享内存访问和同步开销，适合实现高效的 `Reduce`、`Scan` 等操作。

### 2.1 常见接口

```cpp
T __shfl_sync(unsigned mask, T var, int srcLane, int width = warpSize);
T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width = warpSize);
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width = warpSize);
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = warpSize);
```

参数含义：

- `mask`：参与操作的线程掩码，常见写法是 `0xFFFFFFFF`
- `var`：当前线程要交换或参与归约的数据
- `srcLane`：从哪个 lane 读取数据
- `delta`：相对偏移量
- `laneMask`：用于异或交换的掩码
- `width`：参与分组的逻辑宽度，默认是一个 warp

### 2.2 `__shfl_sync`

从指定线程复制数据到当前线程。

```cpp
__global__ void kernel() {
    int val = threadIdx.x;
    int target = __shfl_sync(0xFFFFFFFF, val, 0);
    printf("Thread %d: target=%d\n", threadIdx.x, target);
}
```

用途：

- 广播某个线程的数据
- warp 内共享中间结果

### 2.3 `__shfl_up_sync`

向上取相邻线程的数据，常用于前缀式访问。

```cpp
__global__ void kernel() {
    int val = threadIdx.x;
    int result = __shfl_up_sync(0xFFFFFFFF, val, 2);
    printf("Thread %d: result=%d\n", threadIdx.x, result);
}
```

### 2.4 `__shfl_down_sync`

向下取相邻线程的数据，常用于规约实现。

```cpp
__global__ void kernel() {
    int val = threadIdx.x;
    int result = __shfl_down_sync(0xFFFFFFFF, val, 2);
    printf("Thread %d: result=%d\n", threadIdx.x, result);
}
```

### 2.5 `__shfl_xor_sync`

按照 lane id 的异或结果进行数据交换，常用于 butterfly 风格通信。

```cpp
__global__ void kernel() {
    int val = threadIdx.x;
    int result = __shfl_xor_sync(0xFFFFFFFF, val, 1);
    printf("Thread %d: result=%d\n", threadIdx.x, result);
}
```

### 2.6 Warp Reduce 示例

```cpp
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}
```

Warp 级规约常见于：

- `Softmax`
- `LayerNorm`
- `RMSNorm`
- `Attention`

## 3. 性能分析工具

写 CUDA 代码不能只看“能跑”，还需要判断瓶颈类型。

### 3.1 常见工具

- `ncu`：关注 kernel 级别细节
- `nsys`：关注时间线、调用链和整体执行流程

### 3.2 常见分析维度

- `compute bound`
- `memory bound`
- `bandwidth bound`
- `latency bound`

如果一个 kernel 理论计算量不高，但性能仍差，往往说明问题不在算力，而在访存或调度。

## 4. 访存与共享内存

### 4.1 Bank Conflict

共享内存访问如果映射到同一个 bank，就会发生 bank conflict，导致访问串行化。

可以重点关注：

- 数据排布方式
- 访问步长
- 是否需要 padding
- `swizzling` 机制

### 4.2 向量化加载

可以使用 `float4` 等方式进行向量化加载，提高内存吞吐，但要注意：

- 内存对齐
- 数据布局
- 访存是否连续

## 5. 常见优化手段

### 5.1 MatMul 优化

典型优化方向：

- `tiling`
- 共享内存缓存
- `ping-pong` 双缓冲
- 减少全局内存重复读取

### 5.2 循环展开

```cpp
#pragma unroll
```

循环展开可以减少分支和索引计算开销，但也可能增加寄存器压力，需要结合实际情况评估。

### 5.3 Kernel 融合与重写

某些基础算子在框架默认实现之外，手写 CUDA kernel 往往有明显收益。

常见例子：

- `Softmax`
- `MatMul / GEMM`
- `Transpose`
- `RMSNorm`
- `FlashAttention`

## 6. Softmax 作为典型优化案例

`Softmax` 是 CUDA 优化里非常经典的例子，因为它同时涉及：

- 最大值规约
- 指数计算
- 求和规约
- 数值稳定性

### 6.1 原始公式

给定输入向量：

$$
\mathbf{z} = [z_1, z_2, ..., z_n] \in \mathbb{R}^n
$$

Softmax 定义为：

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

### 6.2 数值稳定写法

为了避免指数溢出，通常会先减去最大值：

$$
\text{Softmax}(z_i) = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^{n} e^{z_j - \max(\mathbf{z})}}
$$

### 6.3 CUDA 实现时的关注点

输入通常可以看成 `N x C` 的形式，需要考虑：

1. 共享内存方案
2. 纯 warpReduce 方案
3. 共享内存 + warpReduce 混合方案

## 7. 后续值得继续展开的专题

后续可以继续把这篇文章扩展成下面几个独立章节：

1. `Softmax` CUDA 实现
2. `MatMul / GEMM` 优化
3. `Transpose` 优化
4. `RMSNorm` 优化
5. `FlashAttention`
6. `CUDA Graph`

## 8. CUDA Graph

`CUDA Graph` 适合在重复执行、调用关系稳定的场景下减少 launch 开销。后续可以单独补充：

- 适用场景
- graph capture 流程
- 与 stream 的关系
- 适合哪些推理任务

