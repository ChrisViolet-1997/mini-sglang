# Session Notes — Phase 1b AliasingGuideTable 实现

## 背景

目标：为 mini-sglang 添加长文本 SQL 推理场景下的 KV Cache 页表别名化优化。
完整方案分三个 Phase，当前在实现 **Phase 1b**：在 tokenize 阶段扫描 token 数组，
找出序列内重复的页对齐 token 块，生成 `AliasingGuideTable`，供 Phase 2 的 KV Cache 别名化使用。

---

## 已完成的改动

### 新建文件

**`python/minisgl/tokenizer/aliasing.py`**
- `AliasEntry(src_page, dst_page, num_pages)` — 描述一个重复块
- `AliasingGuideTable(entries)` — 持有所有重复块信息
- `build_aliasing_guide(input_ids, page_size, min_match_pages=4)` — 核心扫描函数
  - 算法：按 page_size 分页 → 对每页 token 序列计算 Rolling Hash（Rabin-Karp，Mersenne prime）→ 滑动扫描匹配连续 >= min_match_pages 的重复页序列
  - 只记录 dst_page > src_page（非前缀，非自别名）
  - entries 按 dst_page 升序排列

**`tests/misc/test_aliasing.py`**
- 6 个单测覆盖：无重复、单次重复、min_match_pages 阈值、entries 排序、无自别名、序列过短

### 修改的文件

| 文件 | 改动 |
|---|---|
| `python/minisgl/message/backend.py` | import `AliasEntry`, `AliasingGuideTable`；`UserMsg` 加 `aliasing_guide` 字段（带默认值） |
| `python/minisgl/core.py` | `TYPE_CHECKING` 下 import `AliasingGuideTable`；`Req` 加 `aliasing_guide: AliasingGuideTable | None = None` |
| `python/minisgl/scheduler/utils.py` | `PendingReq` 加 `aliasing_guide: AliasingGuideTable | None = None` |
| `python/minisgl/scheduler/prefill.py` | `add_one_req` 透传 `aliasing_guide` 到 `PendingReq`；`_add_one_req` 透传到 `Req` |
| `python/minisgl/tokenizer/tokenize.py` | `TokenizeManager.__init__` 加 `page_size` 参数；`tokenize()` 返回值改为 `List[Tuple[Tensor, AliasingGuideTable]]`；每次 tokenize 后调用 `build_aliasing_guide` |
| `python/minisgl/tokenizer/server.py` | `tokenize_worker` 加 `page_size: int = 1` 参数；实例化 `TokenizeManager(tokenizer, page_size=page_size)`；消费 `tokenize()` 返回的 `(ids, guide)` 元组填入 `UserMsg` |
| `python/minisgl/server/launch.py` | tokenizer 进程的 kwargs 加 `"page_size": server_args.page_size` |

---

## 尚未完成

1. **运行单测验证**：本地 Mac 环境没有合适的 conda env（需要 Python >= 3.10 + torch），测试文件已写好但未跑通。
   - 测试命令：`python -m pytest tests/misc/test_aliasing.py -v --override-ini="addopts="`
   - 安装包：`pip install -e . --no-deps`（再单独装 torch、pytest）

2. **序列化验证**：`AliasingGuideTable` 跨进程经 ZMQ + msgpack 传输的正确性未实测。
   - `message/utils.py` 的 `serialize_type` 用 `__dict__` 递归序列化，理论上支持 dataclass，需运行 `tests/misc/test_serialize.py` 确认。

3. **Phase 2**：KV Cache 页表别名化（修改 `radix_cache.py` 和 `cache.py`），消费 `Req.aliasing_guide`。

---

## 关键设计决策

- **1a（字面量动态压缩）已放弃**：Qwen2.5-Coder 无可用锚点 token，压缩会破坏 SQL 列名/类型语义，影响输出质量。
- `min_match_pages=4` 是最小收益阈值（4 × page_size tokens），避免 overhead 大于收益。
- `AliasingGuideTable` 在 `Req` 上是 `None`-able，存量请求（无重复块）零开销。
- `build_aliasing_guide` 只处理完全对齐的页（`n_tokens // page_size`），尾部不足一页的 token 忽略，与 KV Cache paged attention 语义一致。

---

## 下一步（Phase 2 入口）

- 读 `python/minisgl/scheduler/cache.py` 和 `python/minisgl/kvcache/base.py`
- 找到 `allocate_paged` 的完整调用链
- 在 `_prepare_batch()` 时检查 `req.aliasing_guide`，对命中的重复页做虚拟页 → 物理页的别名映射
- 注意 decode 阶段需要 copy-on-write 保护别名页（别名页只读，新生成 token 写入新页）
