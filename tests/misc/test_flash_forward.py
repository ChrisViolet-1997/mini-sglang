"""
Tests for Phase 3: Decode Flash-Forward.

TestFlashForwardDetector and TestBuildFlashForwardCandidates are pure CPU tests.
TestExecuteFlashForward requires CUDA.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from minisgl.scheduler.flash_forward import (
    FlashForwardCandidate,
    FlashForwardDetector,
    FlashForwardState,
    build_flash_forward_candidates,
    execute_flash_forward,
)
from minisgl.tokenizer.aliasing import AliasEntry, AliasingGuideTable


class TestFlashForwardDetector:
    """Pure CPU tests for the flash-forward FSM detector."""

    def _make_state(self, token_lists, threshold=4):
        candidates = []
        for tl in token_lists:
            t = torch.tensor(tl, dtype=torch.int32)
            candidates.append(
                FlashForwardCandidate(
                    tokens=t,
                    src_start_pos=0,
                    length=len(tl),
                    token_list=tl,
                )
            )
        return FlashForwardState(candidates=candidates, match_threshold=threshold)

    def test_no_match(self):
        """Tokens that don't match any candidate don't trigger."""
        detector = FlashForwardDetector()
        state = self._make_state([[10, 20, 30, 40, 50, 60]])

        for tok in [1, 2, 3, 4, 5, 6, 7, 8]:
            detector.feed_token(state, tok)
            assert detector.get_fast_forward_tokens(state) is None

    def test_partial_match_below_threshold(self):
        """Matching fewer tokens than threshold doesn't trigger."""
        detector = FlashForwardDetector()
        state = self._make_state([[10, 20, 30, 40, 50, 60]], threshold=4)

        # Match 3 tokens (below threshold of 4)
        for tok in [10, 20, 30]:
            detector.feed_token(state, tok)
            assert detector.get_fast_forward_tokens(state) is None

    def test_trigger_at_threshold(self):
        """Matching exactly threshold tokens triggers fast-forward."""
        detector = FlashForwardDetector()
        state = self._make_state([[10, 20, 30, 40, 50, 60]], threshold=4)

        for tok in [10, 20, 30, 40]:
            detector.feed_token(state, tok)

        result = detector.get_fast_forward_tokens(state)
        assert result is not None
        assert result.tolist() == [50, 60]

    def test_remaining_tokens_correct(self):
        """After trigger, remaining tokens exclude the matched prefix."""
        detector = FlashForwardDetector()
        state = self._make_state([[1, 2, 3, 4, 5, 6, 7, 8]], threshold=4)

        for tok in [1, 2, 3, 4]:
            detector.feed_token(state, tok)

        result = detector.get_fast_forward_tokens(state)
        assert result is not None
        assert result.tolist() == [5, 6, 7, 8]

    def test_mismatch_resets(self):
        """A mismatch in the middle resets the state."""
        detector = FlashForwardDetector()
        state = self._make_state([[10, 20, 30, 40, 50]], threshold=4)

        # Match 2, then mismatch
        detector.feed_token(state, 10)
        detector.feed_token(state, 20)
        detector.feed_token(state, 99)  # mismatch

        assert state.active_candidate_idx == -1
        assert state.matched_count == 0
        assert detector.get_fast_forward_tokens(state) is None

    def test_multiple_candidates(self):
        """With multiple candidates, the first matching one is selected."""
        detector = FlashForwardDetector()
        state = self._make_state(
            [[10, 20, 30, 40, 50], [10, 20, 30, 40, 60, 70]],
            threshold=4,
        )

        for tok in [10, 20, 30, 40]:
            detector.feed_token(state, tok)

        # Should match first candidate (idx 0)
        assert state.active_candidate_idx == 0
        result = detector.get_fast_forward_tokens(state)
        assert result is not None
        assert result.tolist() == [50]

    def test_trigger_entire_candidate_no_remaining(self):
        """If all tokens are matched, no remaining tokens to fast-forward."""
        detector = FlashForwardDetector()
        state = self._make_state([[10, 20, 30, 40]], threshold=4)

        for tok in [10, 20, 30, 40]:
            detector.feed_token(state, tok)

        # All 4 tokens matched, nothing remaining
        result = detector.get_fast_forward_tokens(state)
        assert result is None

    def test_reset_clears_state(self):
        """State.reset() clears active matching."""
        detector = FlashForwardDetector()
        state = self._make_state([[10, 20, 30, 40, 50]], threshold=4)

        for tok in [10, 20, 30, 40]:
            detector.feed_token(state, tok)

        state.reset()
        assert state.active_candidate_idx == -1
        assert state.matched_count == 0
        assert detector.get_fast_forward_tokens(state) is None


class TestBuildFlashForwardCandidates:
    """Pure CPU tests for building flash-forward candidates."""

    def test_single_entry(self):
        """Single AliasEntry generates one correct candidate."""
        page_size = 4
        # 32 tokens, pages 0-7
        input_ids = torch.arange(32, dtype=torch.int32)
        guide = AliasingGuideTable(
            entries=[AliasEntry(src_page=1, dst_page=5, num_pages=2)]
        )

        candidates = build_flash_forward_candidates(input_ids, guide, page_size)

        assert len(candidates) == 1
        cand = candidates[0]
        assert cand.src_start_pos == 4  # page 1 * page_size 4
        assert cand.length == 8  # 2 pages * 4 tokens
        assert cand.token_list == list(range(4, 12))

    def test_empty_guide(self):
        """Empty guide produces no candidates."""
        input_ids = torch.arange(32, dtype=torch.int32)
        guide = AliasingGuideTable(entries=[])

        candidates = build_flash_forward_candidates(input_ids, guide, page_size=4)
        assert candidates == []

    def test_multiple_entries(self):
        """Multiple entries produce multiple candidates."""
        page_size = 4
        input_ids = torch.arange(64, dtype=torch.int32)
        guide = AliasingGuideTable(
            entries=[
                AliasEntry(src_page=0, dst_page=4, num_pages=2),
                AliasEntry(src_page=2, dst_page=8, num_pages=3),
            ]
        )

        candidates = build_flash_forward_candidates(input_ids, guide, page_size)

        assert len(candidates) == 2
        # First candidate: src_page=0, 2 pages
        assert candidates[0].src_start_pos == 0
        assert candidates[0].length == 8
        assert candidates[0].token_list == list(range(0, 8))
        # Second candidate: src_page=2, 3 pages
        assert candidates[1].src_start_pos == 8
        assert candidates[1].length == 12
        assert candidates[1].token_list == list(range(8, 20))

    def test_truncates_to_input_length(self):
        """Candidate is truncated if src block extends beyond input_ids."""
        page_size = 4
        input_ids = torch.arange(10, dtype=torch.int32)  # only 10 tokens
        guide = AliasingGuideTable(
            entries=[AliasEntry(src_page=1, dst_page=3, num_pages=4)]
            # src_start=4, length=16, but input only has 10 tokens
        )

        candidates = build_flash_forward_candidates(input_ids, guide, page_size)

        assert len(candidates) == 1
        assert candidates[0].src_start_pos == 4
        assert candidates[0].length == 6  # 10 - 4
        assert candidates[0].token_list == list(range(4, 10))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestExecuteFlashForward:
    """CUDA integration tests for execute_flash_forward."""

    def setup_method(self):
        self.device = torch.device("cuda:0")
        self.page_size = 4
        self.head_dim = 128
        self.num_heads = 8
        self.num_layers = 2
        self.max_position = 4096
        self.max_seq_len = 256
        self.max_reqs = 4
        self.num_pages = 64
        self.kv_dim = self.num_heads * self.head_dim

        # Build cos_sin_cache
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        positions = torch.arange(self.max_position, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(self.device)

        # Build page_table and token_pool
        self.page_table = torch.zeros(
            self.max_reqs, self.max_seq_len, dtype=torch.int32, device=self.device
        )
        self.token_pool = torch.zeros(
            self.max_reqs, self.max_seq_len, dtype=torch.int32, device=self.device
        )

        # Build mock KV cache
        num_physical_tokens = self.num_pages * self.page_size
        self.kv_cache = MagicMock()
        self.kv_cache.num_layers = self.num_layers
        self.kv_cache._storage_shape = (num_physical_tokens, self.num_heads, self.head_dim)
        self.kv_cache._k_buffer = [
            torch.randn(num_physical_tokens, self.num_heads, self.head_dim, device=self.device, dtype=torch.float16)
            for _ in range(self.num_layers)
        ]
        self.kv_cache._v_buffer = [
            torch.randn(num_physical_tokens, self.num_heads, self.head_dim, device=self.device, dtype=torch.float16)
            for _ in range(self.num_layers)
        ]

    def _make_req(self, input_ids_len=16, cached_len=16, output_len=32, table_idx=0):
        """Create a mock Req for testing."""
        req = MagicMock()
        req.input_ids = torch.arange(input_ids_len, dtype=torch.int32)
        req.table_idx = table_idx
        req.device_len = cached_len
        req.cached_len = cached_len
        req.max_device_len = input_ids_len + output_len
        req.uid = 42

        @property
        def remain_len_prop(self):
            return self.max_device_len - self.device_len

        @property
        def can_decode_prop(self):
            return (self.max_device_len - self.device_len) > 0

        type(req).remain_len = remain_len_prop
        type(req).can_decode = can_decode_prop
        return req

    def _setup_page_table_identity(self, table_idx=0, length=64):
        """Set up identity page table mapping (position i -> physical location i)."""
        positions = torch.arange(length, dtype=torch.int32, device=self.device)
        self.page_table[table_idx, :length] = positions

    def _make_cache_manager(self):
        """Create a mock cache manager."""
        cm = MagicMock()
        cm.allocate_paged = MagicMock()
        # Make allocate_paged write identity mapping for new positions
        def mock_allocate(reqs):
            for req in reqs:
                first_pos = req.cached_len
                last_pos = req.device_len
                for pos in range(first_pos, last_pos):
                    self.page_table[req.table_idx, pos] = pos
        cm.allocate_paged.side_effect = mock_allocate
        return cm

    def test_token_pool_written(self):
        """Token pool is correctly written with fast-forwarded tokens."""
        req = self._make_req(input_ids_len=16, cached_len=16, output_len=32, table_idx=0)
        self._setup_page_table_identity(table_idx=0, length=64)
        cache_manager = self._make_cache_manager()

        ff_tokens = torch.tensor([100, 101, 102, 103], dtype=torch.int32)

        execute_flash_forward(
            req=req,
            ff_tokens=ff_tokens,
            src_start_pos=4,
            matched_count=2,
            token_pool=self.token_pool,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            cos_sin_cache=self.cos_sin_cache,
            cache_manager=cache_manager,
            page_size=self.page_size,
        )

        # Check token_pool at positions 16..19
        written = self.token_pool[0, 16:20].cpu()
        assert written.tolist() == [100, 101, 102, 103]

    def test_state_advancement(self):
        """device_len and cached_len are correctly advanced."""
        req = self._make_req(input_ids_len=16, cached_len=16, output_len=32, table_idx=0)
        self._setup_page_table_identity(table_idx=0, length=64)
        cache_manager = self._make_cache_manager()

        ff_tokens = torch.tensor([100, 101, 102, 103], dtype=torch.int32)

        execute_flash_forward(
            req=req,
            ff_tokens=ff_tokens,
            src_start_pos=0,
            matched_count=0,
            token_pool=self.token_pool,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            cos_sin_cache=self.cos_sin_cache,
            cache_manager=cache_manager,
            page_size=self.page_size,
        )

        assert req.device_len == 20  # 16 + 4
        assert req.cached_len == 20  # should equal device_len after execution

    def test_kv_correctness(self):
        """KV copy produces correct values at dst positions."""
        req = self._make_req(input_ids_len=16, cached_len=16, output_len=32, table_idx=0)
        self._setup_page_table_identity(table_idx=0, length=64)
        cache_manager = self._make_cache_manager()

        # Record K values at src positions before the copy
        src_start_pos = 4
        matched_count = 2
        N = 4
        # src positions: 6, 7, 8, 9
        # dst positions: 16, 17, 18, 19

        # Save original K at src
        k_src_before = []
        for layer_id in range(self.num_layers):
            k_buf = self.kv_cache._k_buffer[layer_id].view(-1, self.kv_dim)
            k_src_before.append(k_buf[6:10].clone())

        ff_tokens = torch.tensor([100, 101, 102, 103], dtype=torch.int32)

        execute_flash_forward(
            req=req,
            ff_tokens=ff_tokens,
            src_start_pos=src_start_pos,
            matched_count=matched_count,
            token_pool=self.token_pool,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            cos_sin_cache=self.cos_sin_cache,
            cache_manager=cache_manager,
            page_size=self.page_size,
        )

        # Verify V is directly copied (V is position-independent)
        for layer_id in range(self.num_layers):
            v_buf = self.kv_cache._v_buffer[layer_id].view(-1, self.kv_dim)
            v_src = v_buf[6:10]
            v_dst = v_buf[16:20]
            torch.testing.assert_close(v_dst, v_src, atol=0, rtol=0)

    def test_remain_len_capping(self):
        """Fast-forward tokens are capped to remain_len."""
        # Set up req with only 2 tokens of remain_len
        req = self._make_req(input_ids_len=16, cached_len=16, output_len=2, table_idx=0)
        # max_device_len = 16 + 2 = 18, device_len = 16, remain_len = 2
        self._setup_page_table_identity(table_idx=0, length=64)
        cache_manager = self._make_cache_manager()

        ff_tokens = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
        # Clamp to remain_len before calling execute
        remain = req.max_device_len - req.device_len  # 2
        ff_tokens_clamped = ff_tokens[:remain]

        msgs = execute_flash_forward(
            req=req,
            ff_tokens=ff_tokens_clamped,
            src_start_pos=0,
            matched_count=0,
            token_pool=self.token_pool,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            cos_sin_cache=self.cos_sin_cache,
            cache_manager=cache_manager,
            page_size=self.page_size,
        )

        # Only 2 tokens should be written
        assert len(msgs) == 2
        assert req.device_len == 18  # 16 + 2

    def test_detokenize_msgs_returned(self):
        """Correct DetokenizeMsg list is returned."""
        req = self._make_req(input_ids_len=16, cached_len=16, output_len=32, table_idx=0)
        self._setup_page_table_identity(table_idx=0, length=64)
        cache_manager = self._make_cache_manager()

        ff_tokens = torch.tensor([100, 101, 102], dtype=torch.int32)

        msgs = execute_flash_forward(
            req=req,
            ff_tokens=ff_tokens,
            src_start_pos=0,
            matched_count=0,
            token_pool=self.token_pool,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            cos_sin_cache=self.cos_sin_cache,
            cache_manager=cache_manager,
            page_size=self.page_size,
        )

        assert len(msgs) == 3
        assert msgs[0].uid == 42
        assert msgs[0].next_token == 100
        assert msgs[0].finished is False
        assert msgs[1].next_token == 101
        assert msgs[2].next_token == 102

    def test_empty_ff_tokens(self):
        """Empty ff_tokens is a no-op."""
        req = self._make_req(input_ids_len=16, cached_len=16, output_len=32, table_idx=0)
        self._setup_page_table_identity(table_idx=0, length=64)
        cache_manager = self._make_cache_manager()

        ff_tokens = torch.tensor([], dtype=torch.int32)

        msgs = execute_flash_forward(
            req=req,
            ff_tokens=ff_tokens,
            src_start_pos=0,
            matched_count=0,
            token_pool=self.token_pool,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            cos_sin_cache=self.cos_sin_cache,
            cache_manager=cache_manager,
            page_size=self.page_size,
        )

        assert msgs == []
        assert req.device_len == 16  # unchanged
