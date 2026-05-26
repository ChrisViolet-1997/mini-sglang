<p align="center">
<img width="400" src="/assets/logo.png">
</p>

# Mini-SGLang

A **lightweight yet high-performance** inference framework for Large Language Models.

---

Mini-SGLang is a compact implementation of [SGLang](https://github.com/sgl-project/sglang), designed to demystify the complexities of modern LLM serving systems. With a compact codebase of **~5,000 lines of Python**, it serves as both a capable inference engine and a transparent reference for researchers and developers.

## ✨ Key Features

- **High Performance**: Achieves state-of-the-art throughput and latency with advanced optimizations.
- **Lightweight & Readable**: A clean, modular, and fully type-annotated codebase that is easy to understand and modify.
- **Advanced Optimizations**:
  - **Radix Cache**: Reuses KV cache for shared prefixes across requests.
  - **Chunked Prefill**: Reduces peak memory usage for long-context serving.
  - **Overlap Scheduling**: Hides CPU scheduling overhead with GPU computation.
  - **Tensor Parallelism**: Scales inference across multiple GPUs.
  - **Optimized Kernels**: Integrates **FlashAttention** and **FlashInfer** for maximum efficiency.
  - ...

## 🚀 Quick Start

> **⚠️ Platform Support**: Mini-SGLang currently supports **Linux only** (x86_64 and aarch64). Windows and macOS are not supported due to dependencies on Linux-specific CUDA kernels (`sgl-kernel`, `flashinfer`). We recommend using [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) on Windows or Docker for cross-platform compatibility.

### 1. Environment Setup

We recommend using `uv` for a fast and reliable installation (note that `uv` does not conflict with `conda`).

```bash
# Create a virtual environment (Python 3.10+ recommended)
uv venv --python=3.12
source .venv/bin/activate
```

**Prerequisites**: Mini-SGLang relies on CUDA kernels that are JIT-compiled. Ensure you have the **NVIDIA CUDA Toolkit** installed and that its version matches your driver's version. You can check your driver's CUDA capability with `nvidia-smi`.

### 2. Installation

Install Mini-SGLang directly from the source:

```bash
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
```

<details>
<summary><b>💡 Installing on Windows (WSL2)</b></summary>

Since Mini-SGLang requires Linux-specific dependencies, Windows users should use WSL2:

1. **Install WSL2** (if not already installed):
   ```powershell
   # In PowerShell (as Administrator)
   wsl --install
   ```

2. **Install CUDA on WSL2**:
   - Follow [NVIDIA's WSL2 CUDA guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
   - Ensure your Windows GPU drivers support WSL2

3. **Install Mini-SGLang in WSL2**:
   ```bash
   # Inside WSL2 terminal
   git clone https://github.com/sgl-project/mini-sglang.git
   cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
   uv pip install -e .
   ```

4. **Access from Windows**: The server will be accessible at `http://localhost:8000` from Windows browsers and applications.

</details>

<details>
<summary><b>🐳 Running with Docker</b></summary>

**Prerequisites**:
- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

1. **Build the Docker image**:
   ```bash
   docker build -t minisgl .
   ```

2. **Run the server**:
   ```bash
   docker run --gpus all -p 1919:1919 \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

3. **Run in interactive shell mode**:
   ```bash
   docker run -it --gpus all \
       minisgl --model Qwen/Qwen3-0.6B --shell
   ```

4. **Using Docker Volumes for persistent caches** (recommended for faster subsequent startups):
   ```bash
   docker run --gpus all -p 1919:1919 \
       -v huggingface_cache:/app/.cache/huggingface \
       -v tvm_cache:/app/.cache/tvm-ffi \
       -v flashinfer_cache:/app/.cache/flashinfer \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

</details>

### 3. Online Serving

Launch an OpenAI-compatible API server with a single command.

```bash
# Deploy Qwen/Qwen3-0.6B on a single GPU
python -m minisgl --model "Qwen/Qwen3-0.6B"

# Deploy meta-llama/Llama-3.1-70B-Instruct on 4 GPUs with Tensor Parallelism, on port 30000
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000
```

Once the server is running, you can send requests using standard tools like `curl` or any OpenAI-compatible client.

### 4. Interactive Shell

Chat with your model directly in the terminal by adding the `--shell` flag.

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

![shell-example](https://lmsys.org/images/blog/minisgl/shell.png)

You can also use `/reset` to clear the chat history.

## Flash-Forward: Intra-Sequence KV Reuse for Repetitive Code Generation

This fork extends Mini-SGLang with a three-phase optimization pipeline targeting **long-context SQL/code generation** where output contains repeated token blocks (e.g., `GROUP BY` mirroring `SELECT` fields).

### Results

| Metric | Flash-Forward | Baseline | Speedup |
|--------|--------------|----------|---------|
| Input tokens | 24,758 | 24,758 | — |
| Output tokens | 16,384 | 16,384 | — |
| Prefill (ms) | 55.2 | 47.8 | 0.87x |
| Decode total (ms) | 145,377 | 223,596 | **1.54x** |
| Decode ITI (ms/tok) | 8.87 | 13.65 | **1.54x** |
| Total time (ms) | 145,432 | 223,643 | **1.54x** |
| Throughput (tok/s) | 283 | 184 | **1.54x** |

Environment: RTX 5090 (sm120), Qwen3-8B, page_size=1, flashinfer backend.

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Pre-Prefill (CPU)                                     │
│  Sliding-window hash → AliasingGuideTable (duplicate detection) │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Prefill (GPU)                                         │
│  Two-pass forward: skip duplicate suffix → KV page aliasing     │
│  via copy_kv_with_rope (Triton fused kernel, 10.4x vs PyTorch)  │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Decode (GPU)                                          │
│  FlashForwardDetector (O(1) hash-map FSM) detects repeated      │
│  output → bulk KV copy + token write → skip N autoregressive    │
│  forward passes                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Files

| File | Role |
|------|------|
| `python/minisgl/tokenizer/aliasing.py` | Sliding-window hash, builds `AliasingGuideTable` |
| `python/minisgl/scheduler/kv_alias.py` | `compute_skippable_suffix()`, `apply_kv_aliasing()` |
| `python/minisgl/scheduler/flash_forward.py` | FSM detector + `execute_flash_forward()` |
| `python/minisgl/kernel/copy_kv_rope_triton.py` | Triton fused KV copy with RoPE correction |
| `python/minisgl/kernel/copy_kv_rope.py` | PyTorch fallback for KV copy |
| `benchmark/ab_benchmark.py` | A/B benchmark script |

### Quick Start (Offline with Flash-Forward)

```python
import sys
sys.path.insert(0, "/path/to/mini-sglang/python")

from minisgl.llm import LLM
from minisgl.core import SamplingParams

llm = LLM(
    model_path="Qwen/Qwen3-8B",
    attention_backend="fi",
    page_size=1,
    cuda_graph_max_bs=0,
)

# Flash-forward is automatically enabled when input contains
# intra-sequence duplicate blocks (detected via aliasing guide)
results = llm.generate(
    prompts=["<your long SQL prompt with repeated CASE WHEN blocks>"],
    sampling_params=SamplingParams(max_tokens=16384, temperature=0.0),
)
```

### Running the A/B Benchmark

```bash
python benchmark/ab_benchmark.py
```

Requires a baseline copy at `/root/autodl-tmp/mini-sglang-baseline`. Edit paths in the script as needed.

---

## Benchmark

### Offline inference

See [bench.py](./benchmark/offline/bench.py) for more details. Set `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` for ablation study on overlap scheduling.

Test Configuration:

- Hardware: 1xH200 GPU.
- Model: Qwen3-0.6B, Qwen3-14B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100-1024 tokens
- Output Length: Randomly sampled between 100-1024 tokens

![offline](https://lmsys.org/images/blog/minisgl/offline.png)

### Online inference

See [benchmark_qwen.py](./benchmark/online/bench_qwen.py) for more details.

Test Configuration:

- Hardware: 4xH200 GPU, connected by NVLink.
- Model: Qwen3-32B
- Dataset: [Qwen trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/blob/main/qwen_traceA_blksz_16.jsonl), replaying first 1000 requests.

Launch command:

```bash
# Mini-SGLang
python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --cache naive

# SGLang
python3 -m sglang.launch_server --model "Qwen/Qwen3-32B" --tp 4 \
    --disable-radix --port 1919 --decode-attention flashinfer
```

> **Note**: If you encounter network issues when downloading models from HuggingFace, try using `--model-source modelscope` to download from ModelScope instead:
> ```bash
> python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --model-source modelscope
> ```

![online](https://lmsys.org/images/blog/minisgl/online.png)

## 📚 Learn More

- **[Detailed Features](./docs/features.md)**: Explore all available features and command-line arguments.
- **[System Architecture](./docs/structures.md)**: Dive deep into the design and data flow of Mini-SGLang.
