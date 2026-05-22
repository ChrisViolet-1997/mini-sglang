# Phase 3: Decode Flash-Forward — 实现总结与迁移指南

## 环境要求

- GPU: sm75+ (T4/A100/H100)，V100 (sm70) 不支持
- 依赖:
  - `flash-attn` (已安装)
  - `sgl-kernel` — 需要 sm75+ 的版本，`pip install sgl-kernel`
  - `flashinfer` — 已安装，但 attention kernel 需要 sm75+
- 已添加 PyTorch fallback: RMSNorm / SiLU / RoPE 在 sm70 上自动降级为纯 PyTorch 实现

## 已完成的代码变更

### 新文件

| 文件 | 说明 |
|------|------|
| `python/minisgl/scheduler/flash_forward.py` | Flash-Forward 核心模块 |
| `tests/misc/test_flash_forward.py` | 单元测试 (18 tests, 全部通过) |
| `tests/misc/compare_inference.py` | Baseline vs Optimized 推理对比脚本 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `python/minisgl/core.py` | Req 添加 `_ff_state` 字段 |
| `python/minisgl/scheduler/scheduler.py` | 集成 FlashForwardDetector，post-prefill 初始化，decode 触发逻辑 |
| `python/minisgl/layers/norm.py` | RMSNorm/RMSNormFused 添加 PyTorch fallback (sm70 兼容) |
| `python/minisgl/layers/activation.py` | silu_and_mul/gelu_and_mul 添加 PyTorch fallback |
| `python/minisgl/layers/rotary.py` | RotaryEmbedding 添加 PyTorch RoPE fallback |

## 运行对比测试

迁移到 sm75+ 机器后:

```bash
# 1. 安装依赖
pip install flash-attn --no-build-isolation
pip install sgl-kernel
pip install flashinfer-python

# 2. 运行单元测试
pytest tests/misc/test_flash_forward.py -v
pytest tests/misc/test_kv_alias.py -v

# 3. 运行推理对比 (baseline vs optimized)
python tests/misc/compare_inference.py
```

## 对比脚本说明 (`tests/misc/compare_inference.py`)

- 使用 `input_prompt/demo.txt` 作为输入 (95617 chars, ~46250 tokens 的 Hive SQL)
- Baseline: 标准 LLM 推理，无优化
- Optimized: 附带 AliasingGuideTable，启用 Phase 1/2 prefill 跳过 + Phase 3 decode flash-forward
- 输出: 时间、吞吐量、峰值显存、输出 token 对比
- 使用 greedy decoding (temperature=0) 保证可复现性

## 预期收益 (基于静态分析)

对 demo.txt (46,250 tokens):

| 优化阶段 | 节省量 | 说明 |
|----------|--------|------|
| Phase 1/2 Prefill | 最多 57.1% tokens 跳过 | KV aliasing 避免重复计算 |
| Phase 3 Decode | 最多 9,424 步快进 | 匹配已知重复块时跳过自回归 |
| KV Memory | 57.1% 页面节省 | 别名页共享物理存储 |

## 架构概览

```
Prefill 完成时:
  input_ids + AliasingGuideTable → build_flash_forward_candidates() → [FlashForwardCandidate]
  → FlashForwardState 绑定到 Req._ff_state

Decode 每步:
  sample next_token → detector.feed_token(state, token)
  if state.matched_count >= threshold:
    ff_tokens = get_fast_forward_tokens(state)  # 剩余 tokens
    execute_flash_forward(req, ff_tokens, ...)   # 写 token + copy KV + 推进状态
    state.reset()
  else:
    continue normal decode
```

## 注意事项

1. `compare_inference.py` 中 `OptimizedLLM` 重写了 `offline_receive_msg` 来附加 aliasing guide
2. Page size = 1 时匹配粒度最细，收益最大；实际部署可调整
3. Match threshold 默认 4 tokens，可在 `FlashForwardState` 中调整
4. Flash-forward 不改变模型输出质量 — 只是跳过已知会生成的 token
