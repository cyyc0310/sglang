# MTP + Mixed Chunk Prefill 兼容性 Plans.md

作成日: 2026-05-06

---

## 背景与目标

**目标**: 让 Mixed Chunk Prefill 与 EAGLE V2 (overlap scheduler) 兼容，允许 chunk prefill 请求与 decode 请求的 verify token 在同一个 target forward 中处理。

**当前问题**: spec 解码时 `enable_mixed_chunk` 被强制关闭。已完成的初始修改中，执行顺序为 `draft(decode) → merged target forward(verify+extend) → verify → sample → draft_extend`，prefill 被 draft 多步循环阻塞，TTFT 劣化严重。

**新流程** (Round N, mixed batch = prefill P + decode D):

```
1. 对 D：从存储的 draft 产物 rebuild tree → EagleVerifyInput
   对 P：无 tree，正常 prefill
2. Target forward：一次 forward 处理 P 的 prefill tokens + D 的 verify tokens
3. Sample/Verify：
   - D：rejection sampling → accepted tokens + hidden_states + topk_p/topk_index
   - P：正常 sample → first token + hidden_states + topk_p/topk_index
4. 返回 GenerationBatchResult，scheduler 流式发送 token 给客户端
5. Draft forward：
   - D：用 accepted info 跑 draft → 产出 draft 产物
   - P（刚完成 prefill）：用 first token info 跑 draft → 产出 draft 产物
   - P（还在 chunked prefill）：跳过
6. 存储 draft 产物（per-request）
```

---

## Phase 1: Per-Request Draft 中间产物存储

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 1.1 | 创建 `DraftArtifacts` dataclass | 类定义完成，包含所有必要字段 | - | cc:TODO |
| 1.2 | 在 `Req` 类上添加 `draft_artifacts` 字段 | 字段可读写，默认 None | 1.1 | cc:TODO |
| 1.3 | 修改 `EagleDraftWorker.draft_forward()` 将中间产物写入 per-request 存储 | draft 结束后每个 req 的 draft_artifacts 被填充 | 1.1, 1.2 | cc:TODO |

### Task 1.1 Detail: DraftArtifacts Dataclass

**文件**: `python/sglang/srt/speculative/eagle_info.py`

```python
@dataclass
class DraftArtifacts:
    """Per-request draft intermediate products, stored across rounds."""
    parent_list: torch.Tensor          # 树的 parent index
    top_scores_index: torch.Tensor     # top-k score 的排序索引
    draft_tokens: torch.Tensor         # draft 出的 token IDs
    verified_id: torch.Tensor          # 上一轮 accepted 的最后一个 token
    # 用于下一轮 draft extend
    topk_p: torch.Tensor              # (topk,)
    topk_index: torch.Tensor          # (topk,)
    hidden_states: torch.Tensor       # (hidden_size,)
```

**注意**: 这些 tensor 需要持久化到下一轮。不能是临时 buffer。

### Task 1.2 Detail: Req 字段

**文件**: `python/sglang/srt/managers/schedule_batch.py`

在 `Req` 类中添加:
```python
draft_artifacts: Optional[DraftArtifacts] = None
```

### Task 1.3 Detail: draft_forward 写入存储

**文件**: `python/sglang/srt/speculative/eagle_worker_v2.py` (`EagleDraftWorker.draft_forward`)

在 `draft_forward` 返回后（`forward_batch_generation` 的 decode 分支中），将 `(parent_list, top_scores_index, draft_tokens)` 和对应的 `(topk_p, topk_index, hidden_states)` 写入每个 req 的 `draft_artifacts`。

**关键点**: `draft_forward` 返回的是 batch-level tensor，需要按 request index 拆分到 per-request 存储。

---

## Phase 2: 从存储的 Draft 产物 Rebuild Tree

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 2.1 | 创建 `build_verify_input_from_artifacts()` 函数 | 给定 decode 请求列表，重建 EagleVerifyInput | 1.1 | cc:TODO |
| 2.2 | 创建 `build_mixed_batch_input()` 函数 | 合并 P 的 prefill input 和 D 的 verify input 为统一 forward 输入 | 2.1 | cc:TODO |

### Task 2.1 Detail: Rebuild EagleVerifyInput

**文件**: `python/sglang/srt/speculative/eagle_utils.py` (新函数)

输入: `List[Req]` (decode requests with draft_artifacts)
输出: `EagleVerifyInput`

逻辑:
1. 收集每个 D 请求的 `draft_artifacts`
2. 将 per-request 的 `(parent_list, top_scores_index, draft_tokens, verified_id)` 拼接为 batch-level tensor
3. 调用 `build_tree_kernel_efficient()` 生成 `tree_mask, positions, retrive_index, retrive_next_token, retrive_next_sibling`
4. 构造 `EagleVerifyInput`

**关键挑战**: `build_tree_kernel_efficient` 目前假设所有请求有相同的 `topk` 和 `spec_steps` 参数。per-request 的 `draft_tokens` 长度可能不同（如果 spec_steps/topk 不同...但实际上是固定的）。需要处理 batch 化拼接。

### Task 2.2 Detail: 合并 Mixed Batch Input

**文件**: `python/sglang/srt/speculative/eagle_worker_v2.py` (新方法)

需要为 target forward 构建统一的 ForwardBatch:
1. **P 请求**: `input_ids` = prefill tokens, `extend_seq_lens` = 各请求的 extend 长度, 无 custom_mask
2. **D 请求**: `input_ids` = draft_tokens (flattened), `extend_seq_lens` = speculative_num_draft_tokens per req, `custom_mask` = tree_mask

合并方式:
- `input_ids = cat([P_input_ids, D_input_ids])`
- `extend_seq_lens = cat([P_extend_lens, D_verify_lens])`
- `custom_mask`: 块对角矩阵
  - P 部分: 标准 causal mask（或不设 custom_mask，使用默认 causal）
  - D 部分: tree attention mask
- `positions`: 分别计算
- `out_cache_loc`: 分别分配 KV cache slot

**注意**: 这是最复杂的部分。需要深入理解 attention backend 如何处理 custom_mask。

---

## Phase 3: 重排 V2 Worker 执行流

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 3.1 | 修改 `EAGLEWorkerV2.forward_batch_generation()` 添加 MIXED 分支 | MIXED mode 走新流程，EXTEND/DECODE 不受影响 | 2.2 | cc:TODO |
| 3.2 | 实现 combined target forward (prefill + verify) | 一次 GPU forward 处理两类 token | 3.1 | cc:TODO |
| 3.3 | 实现 split sample/verify 逻辑 | P 正常 sample，D rejection sampling | 3.2 | cc:TODO |
| 3.4 | 实现 post-verify draft forward | D 和新完成 prefill 的 P 都跑 draft | 3.3 | cc:TODO |
| 3.5 | 存储 draft 产物到 per-request Req | Step 6 的存储逻辑 | 3.4, 1.2 | cc:TODO |

### Task 3.1 Detail: 新的 forward_batch_generation 流程

**文件**: `python/sglang/srt/speculative/eagle_worker_v2.py`

当前结构 (`line 634`):
```python
def forward_batch_generation(self, model_worker_batch):
    if is_extend or is_extend_in_batch:
        # EXTEND path: target prefill + draft prefill
    else:
        # DECODE path: draft → verify → draft_extend
```

新增 MIXED 分支:
```python
def forward_batch_generation(self, model_worker_batch):
    if model_worker_batch.forward_mode.is_mixed():
        # MIXED path (NEW):
        # 1. Build verify input from stored artifacts (D requests only)
        # 2. Combined target forward (P prefill + D verify)
        # 3. Split sample/verify
        # 4. Return GenerationBatchResult
        # 5. Draft forward (D + newly-prefilled P)
        # 6. Store draft artifacts
    elif is_extend or is_extend_in_batch:
        # EXTEND path (unchanged)
    else:
        # DECODE path (unchanged)
```

### Task 3.2 Detail: Combined Target Forward

核心挑战: 在一次 forward 中处理两种 attention pattern。

**方案 A (推荐): 统一使用 custom_mask**
- 为整个 batch 构建统一的 custom_mask
- P 部分: 构建标准 causal mask block（lower triangular）
- D 部分: 使用 tree attention mask
- 所有 token 都通过 `ForwardMode.TARGET_VERIFY` 或新的 `ForwardMode.MIXED_VERIFY` 模式

**方案 B: 分别 forward**
- P 走 `ForwardMode.EXTEND`
- D 走 `ForwardMode.TARGET_VERIFY`
- 两次 forward 但可以在不同 stream 上 overlap
- 缺点: 两次 forward 增加延迟，但实现更简单

**方案 A 的实现细节**:
- `custom_mask` shape: `(total_tokens, max_seq_len)` 或使用现有 verify mask 格式
- P 的 causal mask 可以用现有的 `compute_attention_mask_extend` 逻辑预计算
- D 的 tree mask 直接用 `build_tree_kernel_efficient` 产出
- 拼接为 block-diagonal mask

### Task 3.3 Detail: Split Sample/Verify

target forward 返回统一的 logits_output:
- P 部分的 logits: 正常 sample → `next_token_ids[i]`, `hidden_states[i]`
- D 部分的 logits: 使用 `verify_input.sample()` 做 rejection sampling → `verified_id`, `accept_length`

需要:
1. 按 P/D 分割 logits output
2. P 部分: `sample() → (next_token_ids, topk_p, topk_index, hidden_states)`
3. D 部分: `rejection_sampling() → (accepted_tokens, accept_length, verified_id, topk_p, topk_index, hidden_states)`
4. 合并结果为统一的 `GenerationBatchResult`

### Task 3.4 Detail: Post-Verify Draft Forward

分三组处理:
1. **D (decode)**: 用 accepted info 跑 draft → 类似现有 `_draft_extend_for_decode`
2. **P-done (刚完成 prefill 的)**: 用 first token info 跑 draft → 类似现有 `_draft_extend_for_prefill`
3. **P-chunked (还在 chunked prefill 的)**: 跳过，不跑 draft

**实现**:
- 将 D 和 P-done 合并为一个 draft batch
- P-chunked 的 draft_artifacts 保持 None
- Draft forward 结束后，将产物写入各自的 `req.draft_artifacts`

### Task 3.5 Detail: 存储 Draft Artifacts

在 draft forward 完成后:
```python
for i, req in enumerate(decode_reqs + newly_prefilled_reqs):
    req.draft_artifacts = DraftArtifacts(
        parent_list=parent_list[i],
        top_scores_index=top_scores_index[i],
        draft_tokens=draft_tokens[i],
        verified_id=verified_id[i],
        topk_p=topk_p[i],
        topk_index=topk_index[i],
        hidden_states=hidden_states[i],
    )
```

---

## Phase 4: Scheduler 集成

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 4.1 | 移除 `enable_mixed_chunk` + spec 互斥 guard | 两者可同时启用 | - | cc:TODO |
| 4.2 | 修改 overlap scheduler 处理 MIXED + spec batch | process_batch_result 正确处理混合结果 | 3.1 | cc:TODO |
| 4.3 | Prefill 结果提前返回 | TTFT 不受 draft 阻塞 | 4.2 | cc:TODO |
| 4.4 | `process_batch_result()` 处理 MIXED + spec 的结果路由 | decode 和 prefill 结果分别处理 | 4.2 | cc:TODO |

### Task 4.1 Detail: 移除互斥 Guard

**文件**: `python/sglang/srt/server_args.py`

需要修改的 guards:
1. **Line ~2234-2239**: eagle spec 时强制关闭 mixed chunk
   - 移除 `enable_mixed_chunk = False` 的赋值
2. **Line ~4916-4919**: 最终 assertion
   - 修改为条件检查（仅 V1 不支持？或者完全移除）

### Task 4.2 Detail: Overlap Scheduler MIXED + Spec

**文件**: `python/sglang/srt/managers/scheduler.py`

`run_batch()` 中 spec V2 分支 (`line 2271-2304`):
- 当前: 只处理 DECODE 或 EXTEND
- 新增: 处理 MIXED mode
  - `batch.spec_info` 需要包含 D 的 verify input
  - `batch.seq_lens` 更新需要区分 P 和 D

### Task 4.3 Detail: Prefill 提前返回

在 `process_batch_result()` 中，MIXED batch 的结果处理:
1. P 请求: 流式发送 generated tokens → client，将 P 加入 decode 队列
2. D 请求: 发送 accepted tokens → client，filter finished

P 的 token 可以在 step 3 (sample) 完成后立即返回，不需要等 draft forward 完成。这在 scheduler 的 `pop_and_process()` 中自然实现——forward stream 完成后就可以开始处理结果。

### Task 4.4 Detail: 结果路由

**文件**: `python/sglang/srt/managers/scheduler_output_processor_mixin.py`

当前 `process_batch_result_prefill()` 处理 MIXED batch，但不处理 spec 逻辑。

需要修改:
- 识别 batch 中的 P 和 D 请求（通过 `decoding_reqs` 或新字段）
- P 请求: 走正常 prefill result 处理
- D 请求: 走 spec decode result 处理（accept_length, verified_id 等）

---

## Phase 5: 测试与验证

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 5.1 | 单元测试: DraftArtifacts 存储与读取 | 存取正确，tensor shape 匹配 | 1.3 | cc:TODO |
| 5.2 | 单元测试: build_verify_input_from_artifacts | 重建的 tree 与直接 build 的结果一致 | 2.1 | cc:TODO |
| 5.3 | 集成测试: mixed batch spec forward | forward 产出正确 token | 3.4 | cc:TODO |
| 5.4 | TTFT 基准测试 | mixed spec 的 TTFT 接近无 spec 的 mixed chunk | 4.4 | cc:TODO |

---

## 关键技术风险

### R1: Combined Attention Mask (高优先级)

**问题**: P 的 prefill tokens 和 D 的 verify tokens 需要不同的 attention pattern。当前 attention backend 不支持在同一个 batch 中混合 causal mask 和 tree mask。

**缓解方案**:
- 方案 A: 为 P 构造显式 causal mask，与 D 的 tree mask 拼接为统一 custom_mask → 需要修改 attention backend
- 方案 B: 分两次 forward，但用 CUDA stream overlap → 实现简单但 GPU 利用率低
- 方案 C: 将 P 的 prefill 也套入 tree 结构（trivial tree: 只有一条路径）→ 复用现有 tree attention 路径

**建议**: 先用方案 B 验证流程正确性，再优化为方案 A 或 C。

### R2: Per-Request Draft 状态管理 (中优先级)

**问题**: 当前 draft 状态是 batch-level 的（`batch.spec_info`），不与具体请求绑定。改为 per-request 存储需要较大的架构调整。

**缓解方案**: 先在 `Req` 上存储 `DraftArtifacts`，draft forward 时从 `Req` 读取并拼接为 batch-level tensor。这是最直接的方案。

### R3: Overlap Scheduler 同步 (中优先级)

**问题**: V2 overlap scheduler 使用 `FutureMap` + `verify_done` event 进行同步。mixed batch 增加了新的同步点（P 不需要 verify_done）。

**缓解方案**: 保持 D 的 verify_done 机制不变，P 的 draft 产物在 draft forward 完成后直接存储到 `Req.draft_artifacts`，不走 `FutureMap`。

### R4: Chunked Prefill 中间状态 (低优先级)

**问题**: 如果 P 是 chunked prefill（还没完成全部 prefill），P 不能跑 draft。下一轮 P 可能继续 prefill（EXTEND）也可能完成 prefill 进入 DECODE。

**缓解方案**: P 的 `draft_artifacts` 保持 None 直到 P 完成 prefill。完成后下一轮走 `_draft_extend_for_prefill` 路径初始化 draft 状态。

---

## 实现优先级建议

1. **Phase 1** (存储基础) → 最先实现，无破坏性
2. **Phase 2 Task 2.1** (tree rebuild) → 验证存储方案可行性
3. **Phase 3 Task 3.1** (MIXED 分支框架) → 先用方案 B（两次 forward）验证端到端流程
4. **Phase 4** (scheduler 集成) → 端到端可运行
5. **Phase 3 Task 3.2** (combined forward) → 性能优化，改为方案 A/C
6. **Phase 5** (测试) → 贯穿整个开发过程
