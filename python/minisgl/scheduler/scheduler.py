from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .flash_forward import (
    FlashForwardDetector,
    FlashForwardState,
    build_flash_forward_candidates,
    execute_flash_forward,
)
from .io import SchedulerIOMixin
from .kv_alias import apply_kv_aliasing, compute_skippable_suffix
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        self.page_size = config.page_size
        # self.config = config

        # Retrieve cos_sin_cache for KV aliasing RoPE correction
        self._cos_sin_cache = self._get_cos_sin_cache(config)

        # Flash-forward detector for decode-time fast-forwarding
        self.ff_detector = FlashForwardDetector()

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    @staticmethod
    def _get_cos_sin_cache(config: SchedulerConfig) -> torch.Tensor:
        """Retrieve the precomputed cos_sin_cache from the rotary embedding layer."""
        from minisgl.layers.rotary import get_rope

        rc = config.model_config.rotary_config
        rope = get_rope(
            head_dim=rc.head_dim,
            rotary_dim=rc.rotary_dim,
            max_position=rc.max_position,
            base=rc.base,
            rope_scaling=tuple(rc.scaling.items()) if rc.scaling else None,
        )
        cache = rope._cos_sin_cache
        if not cache.is_cuda:
            cache = cache.to(torch.device(f"cuda:{config.tp_info.rank}"))
        return cache

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)
                    # Post-prefill: initialize flash-forward state
                    if req.aliasing_guide and not req.aliasing_guide.empty:
                        candidates = build_flash_forward_candidates(
                            req.input_ids, req.aliasing_guide, self.page_size
                        )
                        if candidates:
                            req._ff_state = FlashForwardState(candidates=candidates)
                elif not finished and req._ff_state is not None:
                    # Decode phase: check for flash-forward trigger
                    self.ff_detector.feed_token(req._ff_state, next_token)
                    ff_tokens = self.ff_detector.get_fast_forward_tokens(req._ff_state)
                    if ff_tokens is not None and len(ff_tokens) > 0:
                        # Clamp to remain_len
                        ff_tokens = ff_tokens[: req.remain_len]
                        if len(ff_tokens) > 0:
                            cand = req._ff_state.candidates[req._ff_state.active_candidate_idx]
                            ff_msgs = execute_flash_forward(
                                req=req,
                                ff_tokens=ff_tokens,
                                src_start_pos=cand.src_start_pos,
                                matched_count=req._ff_state.matched_count,
                                token_pool=self.token_pool,
                                page_table=self.engine.page_table,
                                kv_cache=self.engine.kv_cache,
                                cos_sin_cache=self._cos_sin_cache,
                                cache_manager=self.cache_manager,
                                page_size=self.page_size,
                            )
                            reply.extend(ff_msgs)
                            req._ff_state.reset()
                            # Re-check finish after fast-forward
                            if not req.can_decode:
                                self.decode_manager.remove_req(req)
                                self._free_req_resources(req)
                                new_finished_reqs.add(req)
                                # Mark last ff msg as finished
                                if ff_msgs:
                                    ff_msgs[-1] = DetokenizeMsg(
                                        uid=req.uid,
                                        next_token=ff_msgs[-1].next_token,
                                        finished=True,
                                    )

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        if batch.is_prefill:
            for req in batch.reqs:
                req._skip_suffix_len = compute_skippable_suffix(req, self.page_size)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        has_skip = batch.is_prefill and any(
            r._skip_suffix_len > 0 for r in batch.reqs
        )

        if not has_skip:
            # V1 path: single forward + post-copy
            batch.input_ids = self.token_pool[input_mapping]
            forward_output = self.engine.forward_batch(batch, sample_args)
            self.token_pool[output_mapping] = forward_output.next_tokens_gpu
            if batch.is_prefill:
                apply_kv_aliasing(
                    batch, self.engine.kv_cache, self.engine.page_table,
                    self._cos_sin_cache, self.page_size,
                )
            self.decode_manager.filter_reqs(forward_input.batch.reqs)
            return forward_output
        else:
            # V1.1 two-pass path
            return self._forward_two_pass(forward_input)

    def _forward_two_pass(self, forward_input: ForwardInput) -> ForwardOutput:
        """Two-pass forward: pass 1 (bulk non-dst), KV copy, pass 2 (last token)."""
        batch, sample_args, input_mapping, output_mapping = forward_input

        # --- Pass 1: forward non-dst extend tokens ---
        batch.input_ids = self.token_pool[input_mapping]
        with self.engine.ctx.forward_batch(batch):
            self.engine.model.forward()

        # --- KV Copy: fill dst positions from src ---
        apply_kv_aliasing(
            batch, self.engine.kv_cache, self.engine.page_table,
            self._cos_sin_cache, self.page_size,
        )

        # --- Pass 2: forward last token per request with full context ---
        self._prepare_pass2(batch)
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output

    def _prepare_pass2(self, batch: Batch) -> None:
        """Rebuild batch metadata for pass 2: one token per request with full context."""
        page_table = self.engine.page_table
        device = self.device

        # Positions: last position (device_len - 1) for each real req
        positions_list = []
        input_ids_list = []
        out_loc_list = []

        for req in batch.padded_reqs:
            last_pos = req.device_len - 1
            positions_list.append(last_pos)
            input_ids_list.append(self.token_pool[req.table_idx, last_pos])
            out_loc_list.append(int(page_table[req.table_idx, last_pos].item()))

        batch.positions = torch.tensor(
            positions_list, dtype=torch.int32, device=device
        )
        batch.input_ids = torch.stack(input_ids_list)
        batch.out_loc = torch.tensor(out_loc_list, dtype=torch.int32, device=device)

        # Prepare decode-like metadata with full context
        self.engine.attn_backend.prepare_metadata_pass2(batch)


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.forward_extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.forward_extend_len
        torch.arange(
            req.cached_len,
            req.cached_len + length,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.forward_extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
