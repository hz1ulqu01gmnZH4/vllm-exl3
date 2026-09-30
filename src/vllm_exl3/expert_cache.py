"""Packed EXL3 GPU expert cache and shared prefill staging.

Explicit, single-device prototype API. Host sources own all expert bytes; GPU
slots are disposable copies. The planner runs on the GPU and EXL3's logical
expert pointer tables are updated in place, including during graph replay.
This module does not install a checkpoint loader or change a vLLM launcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from . import expert_cache_tables as policy

PROJECTIONS = ("gate", "up", "down")
ATTRIBUTES = ("trellis", "suh", "svh")
MAX_DECODE_LANES = 64


@dataclass(frozen=True)
class Segment:
    name: str
    offset: int  # bytes within a packed expert row
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int


@dataclass
class PackedExpertLayer:
    """Pinned, row-contiguous sources with one precision per layer."""

    data: torch.Tensor  # [experts, words], int32, pinned host allocation
    segments: tuple[Segment, ...]
    bits: int
    hidden: int
    intermediate: int
    shared_suh: tuple[bool, ...]

    @classmethod
    def from_experts(cls, experts: list[dict[str, dict[str, torch.Tensor]]]):
        """Pack complete mul1 experts without dequantizing or changing bits."""
        if not experts:
            raise ValueError("At least one complete EXL3 expert is required")
        first = experts[0]
        hidden = first["gate"]["suh"].numel()
        intermediate = first["gate"]["svh"].numel()
        if hidden % 128 or intermediate % 128:
            raise ValueError("EXL3 cache requires 128-aligned expert dimensions")
        bits = first["gate"]["trellis"].shape[-1] // 16
        if bits not in (2, 3):
            raise ValueError("This cache prototype supports K2/K3 mul1 experts")
        segments = []
        offset = 0
        for projection in PROJECTIONS:
            in_dim, out_dim = (
                (intermediate, hidden) if projection == "down"
                else (hidden, intermediate)
            )
            shapes = {
                "trellis": (in_dim // 16, out_dim // 16, bits * 16),
                "suh": (in_dim,), "svh": (out_dim,),
            }
            for attr in ATTRIBUTES:
                t = first[projection][attr]
                dtype = torch.int16 if attr == "trellis" else torch.float16
                if tuple(t.shape) != shapes[attr] or t.dtype != dtype:
                    raise ValueError(f"Invalid {projection}.{attr} geometry/dtype")
                nbytes = t.numel() * t.element_size()
                segments.append(Segment(
                    f"{projection}_{attr}", offset, tuple(t.shape), dtype, nbytes
                ))
                offset += nbytes
        if offset % 4:
            raise ValueError("Packed expert rows must be word-aligned")
        data = torch.empty(
            (len(experts), offset // 4), dtype=torch.int32,
            device="cpu", pin_memory=True,
        )
        shared_suh = []
        for e, expert in enumerate(experts):
            for projection in PROJECTIONS:
                tensors = expert[projection]
                if set(tensors) != {"trellis", "suh", "svh", "mul1"}:
                    raise ValueError("Expected exactly trellis/suh/svh/mul1; biases unsupported")
                marker = tensors["mul1"]
                if marker.numel() != 1 or int(marker) != -2082680531:
                    raise ValueError("EXL3 cache requires the mul1 codebook marker")
            for segment in segments:
                projection, attr = segment.name.split("_", 1)
                t = expert[projection][attr]
                if tuple(t.shape) != segment.shape or t.dtype != segment.dtype:
                    raise ValueError("Expert shapes/precision must be uniform within a layer")
                target = data[e].view(torch.uint8)[
                    segment.offset:segment.offset + segment.nbytes
                ]
                target.copy_(t.detach().contiguous().view(torch.uint8).reshape(-1))
            shared_suh.append(torch.equal(
                expert["gate"]["suh"], expert["up"]["suh"]
            ))
        return cls(data, tuple(segments), bits, hidden, intermediate, tuple(shared_suh))

    @property
    def num_experts(self):
        return self.data.shape[0]

    def view(self, bank, expert, segment):
        raw = bank[expert].view(torch.uint8)
        return raw[segment.offset:segment.offset + segment.nbytes].view(
            segment.dtype
        ).view(segment.shape)

    def inners(self, bank):
        """Views for the plugin's staged-prefill GEMMs; addresses never move."""
        result = []
        for expert in range(self.num_experts):
            pack = {}
            for projection in PROJECTIONS:
                attrs = {
                    s.name.split("_", 1)[1]: self.view(bank, expert, s)
                    for s in self.segments if s.name.startswith(projection + "_")
                }
                pack[projection] = SimpleNamespace(
                    **attrs, K=self.bits, mcg=False, mul1=True,
                    in_features=attrs["suh"].numel(),
                    out_features=attrs["svh"].numel(),
                )
            pack["_exl3_gate_up_shared_suh"] = self.shared_suh[expert]
            result.append(pack)
        return result


@triton.jit
def _update_pointers(
    mapping, offsets, pointers, bank_address,
    ROW_BYTES: tl.constexpr, EXPERTS: tl.constexpr, BLOCK: tl.constexpr,
):
    expert = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    segment = tl.program_id(1)
    slot = tl.load(mapping + expert, expert < EXPERTS, other=0).to(tl.int64)
    address = bank_address + tl.maximum(slot, 0) * ROW_BYTES + tl.load(offsets + segment)
    tl.store(pointers + segment * EXPERTS + expert, address, expert < EXPERTS)


@triton.jit
def _copy_misses(
    src, dst, src_rows, dst_rows, count_ptr,
    SRC_WORDS: tl.constexpr, DST_WORDS: tl.constexpr,
    PROGRAMS: tl.constexpr, BLOCK: tl.constexpr,
):
    count = tl.load(count_ptr)
    chunks: tl.constexpr = triton.cdiv(SRC_WORDS, BLOCK)
    for item in range(tl.program_id(0), count * chunks, PROGRAMS):
        pair = item // chunks
        chunk = item % chunks
        s = tl.load(src_rows + pair).to(tl.int64)
        d = tl.load(dst_rows + pair).to(tl.int64)
        col = chunk * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(src + s * SRC_WORDS + col, col < SRC_WORDS, other=0)
        tl.store(dst + d * DST_WORDS + col, values, col < SRC_WORDS)


class Exl3ExpertCache:
    """LRU expert banks, optionally partitioned by packed row size.

    Precision pools remove K2 padding and retain cross-layer LRU within each pool.

    Calls must be serialized on one CUDA stream, as in the vLLM model runner.
    Call set_promotions(False) before profiling/capture and True afterwards.
    Host sources must remain immutable for the lifetime of this object.
    """

    def __init__(self, sources, *, slots_per_layer, top_k, max_decode_tokens=4,
                 device="cuda:0", cache_policy="global-lru"):
        from .exl3 import TEMP_ROWS_FUSED, _exl3_moe_arity, load_exllamav3_ext

        self.device = torch.device(device)
        if self.device.type != "cuda" or not sources:
            raise ValueError("EXL3 expert cache requires CUDA and nonempty sources")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if isinstance(slots_per_layer, int):
            slots_per_layer = [slots_per_layer] * len(sources)
        if len(slots_per_layer) != len(sources):
            raise ValueError("Expected a slot count for every source layer")
        first = sources[0]
        for source in sources:
            if (source.num_experts, source.hidden, source.intermediate) != (
                first.num_experts, first.hidden, first.intermediate
            ):
                raise ValueError("All cached layers must share expert geometry")
            if source.data.device.type != "cpu" or not source.data.is_pinned():
                raise ValueError("Expert sources must be pinned host tensors")
            if source.data.dtype != torch.int32 or not source.data.is_contiguous():
                raise ValueError("Expert sources must be contiguous int32 rows")
        if not 0 < top_k <= first.num_experts or max_decode_tokens < 1:
            raise ValueError("Invalid routing geometry")
        if top_k * max_decode_tokens > MAX_DECODE_LANES:
            raise ValueError("Decode routes exceed the 64-lane planner limit")
        if any(not top_k <= s < first.num_experts for s in slots_per_layer):
            raise ValueError("Slots per layer must be >= top_k and < expert count")
        if cache_policy not in ("global-lru", "precision-lru"):
            raise ValueError(f"Unknown EXL3 cache policy: {cache_policy}")
        self.cache_policy = cache_policy
        self.ext = load_exllamav3_ext()
        if _exl3_moe_arity(self.ext.exl3_moe) != 35:
            raise RuntimeError("Cache prototype requires the ExLlamaV3 1.5.x MoE ABI")
        if not hasattr(self.ext, "pinned_cuda_view"):
            raise RuntimeError("ExLlamaV3 must provide pinned_cuda_view for WSL-safe UVA")
        self.sources = list(sources)
        self.top_k = top_k
        self.max_decode_tokens = max_decode_tokens
        self.hidden = first.hidden
        self.num_experts = first.num_experts
        staging = top_k * max_decode_tokens
        self.tables = policy.allocate_global_tables(
            self.device, self.num_experts, slots_per_layer, staging
        )
        self.policy_tables = {}
        self.layer_tables = []
        self.row_words = max(s.data.shape[1] for s in sources)
        self.banks = {}
        if cache_policy == "global-lru":
            self.bank = torch.zeros(
                (sum(slots_per_layer) + staging, self.row_words),
                dtype=torch.int32, device=self.device,
            )
            self.banks[0] = self.bank
            self.policy_tables[0] = self.tables
        else:
            for words in sorted({s.data.shape[1] for s in sources}):
                rows = sum(n for s, n in zip(sources, slots_per_layer)
                           if s.data.shape[1] == words)
                self.banks[words] = torch.zeros(
                    (rows + staging, words), dtype=torch.int32, device=self.device,
                )
                if cache_policy == "precision-lru":
                    table = policy.allocate_global_tables(
                        self.device, self.num_experts,
                        [n for s, n in zip(sources, slots_per_layer)
                         if s.data.shape[1] == words], staging,
                    )
                    table.gate = self.tables.gate
                    self.policy_tables[words] = table
        self.layer_banks = []
        self.prefill_bank = torch.empty(
            (self.num_experts, self.row_words), dtype=torch.int32, device=self.device
        )
        self.host_views = [
            self.ext.pinned_cuda_view(s.data, self.device.index) for s in sources
        ]
        concurrency = self.ext.exl3_moe_max_concurrency(self.device.index)
        if concurrency < 1:
            raise RuntimeError("Invalid EXL3 fused-kernel concurrency")
        self.temps = tuple(
            torch.empty((concurrency, TEMP_ROWS_FUSED, dim),
                        dtype=torch.float16, device=self.device)
            for dim in (first.hidden, first.hidden, first.intermediate, first.intermediate)
        )
        self.buffers = []
        self.states = []
        self.prefill_states = []
        self.pointer_tables = []
        self.segment_offsets = []
        start = 0
        starts = dict.fromkeys(self.banks, 0)
        group_layers = dict.fromkeys(self.banks, 0)
        for layer_index, (source, slots) in enumerate(zip(sources, slots_per_layer)):
            if cache_policy != "global-lru":
                key = source.data.shape[1]
                bank = self.banks[key]
                start = starts[key]
                self.layer_tables.append((self.policy_tables[key], group_layers[key]))
                group_layers[key] += 1
                starts[key] += slots
            else:
                bank = self.bank
                self.layer_tables.append((self.tables, layer_index))
            self.layer_banks.append(bank)
            bank[start:start + slots, :source.data.shape[1]].copy_(
                source.data[:slots], non_blocking=True
            )
            start += slots
            self.buffers.append(policy.allocate_step_buffers(
                self.device, self.num_experts, triton.next_power_of_2(staging)
            ))
            prefill_ptrs = {
                s.name: self.prefill_bank.data_ptr()
                + torch.arange(self.num_experts, dtype=torch.int64, device=self.device)
                * self.row_words * 4 + s.offset for s in source.segments
            }
            attrs = {
                "_exl3_fused_temps": self.temps, "_exl3_k": source.bits,
                "_exl3_codebook_flags": (False, True) * 3,
                "_exl3_inners": source.inners(self.prefill_bank),
            }
            pointer_table = torch.empty(
                (len(source.segments), self.num_experts), dtype=torch.int64,
                device=self.device,
            )
            self.pointer_tables.append(pointer_table)
            self.segment_offsets.append(torch.tensor(
                [s.offset for s in source.segments], dtype=torch.int64, device=self.device,
            ))
            self.states.append(SimpleNamespace(**attrs, _exl3_ptrs={
                segment.name: pointer_table[j] for j, segment in enumerate(source.segments)
            }))
            self.prefill_states.append(SimpleNamespace(**attrs, _exl3_ptrs=prefill_ptrs))
        policy.set_control(self.tables, gate=0)

    def set_promotions(self, enabled):
        """Freeze during graph capture; misses still execute from staging slots."""
        policy.set_gate(self.tables, enabled)

    def attach(self, layers):
        """Bind explicit post-load layers to the plugin's normal expert dispatch."""
        if len(layers) != len(self.sources):
            raise ValueError("Layer/source count mismatch")
        for layer, source in zip(layers, self.sources):
            if getattr(layer, "_exl3_expert_cache", None) is not None:
                raise ValueError("Layer already has an EXL3 cache")
            if getattr(layer, "expert_map", None) is not None:
                raise ValueError("Cache prototype requires whole experts at TP=EP=1")
            if (layer._exl3_hidden_size, layer._exl3_intermediate_local) != (
                source.hidden, source.intermediate
            ):
                raise ValueError("Layer/source expert geometry mismatch")
            if layer._exl3_bits != source.bits:
                raise ValueError("Layer/source precision mismatch")
        for i, layer in enumerate(layers):
            layer._exl3_expert_cache = (self, i)

    def apply(self, layer_index, x, ids, weights, *, limit=None):
        """Execute routed experts; invalid inputs fail before address remapping."""
        from .exl3 import apply_exl3_fused_moe, get_moe_kernel_backend

        if get_moe_kernel_backend() != "exllamav3":
            raise RuntimeError("Set VLLM_EXL3_MOE_KERNEL=exllamav3 for the expert cache")
        if not 0 <= layer_index < len(self.sources):
            raise ValueError("Invalid cache layer index")
        if (x.ndim != 2 or x.shape[1] != self.hidden or x.shape[0] < 1
                or x.dtype not in (torch.float16, torch.bfloat16)
                or ids.shape != (x.shape[0], self.top_k)
                or ids.dtype not in (torch.int32, torch.int64)
                or weights.shape != ids.shape or not weights.is_floating_point()
                or any(t.device != self.device for t in (x, ids, weights))):
            raise ValueError("Invalid EXL3 cache input/routing shape, dtype or device")
        if not ids.is_contiguous():
            raise ValueError("Cache routing IDs must be contiguous")
        valid = (ids >= -1) & (ids < self.num_experts)
        finite = torch.isfinite(weights) & (weights >= 0)
        torch._assert_async((valid & (finite | (ids == -1))).all(), "Invalid EXL3 routes")
        sorted_ids = ids.sort(dim=-1).values
        unique = (sorted_ids[:, 1:] != sorted_ids[:, :-1]) | (sorted_ids[:, 1:] == -1)
        torch._assert_async(unique.all(), "EXL3 requires distinct experts within each token")
        source = self.sources[layer_index]
        if x.shape[0] <= self.max_decode_tokens:
            buffers = self.buffers[layer_index]
            bank = self.layer_banks[layer_index]
            table, index = self.layer_tables[layer_index]
            policy.step(table, index, ids, buffers)
            torch._assert_async((table.error == 0).all(), "EXL3 cache planner failed")
            _copy_misses[(32,)](
                self.host_views[layer_index], bank,
                buffers.gather_src, buffers.gather_dst, buffers.gather_count,
                SRC_WORDS=source.data.shape[1], DST_WORDS=bank.shape[1],
                PROGRAMS=32, BLOCK=4096,
            )
            state = self.states[layer_index]
            _update_pointers[(triton.cdiv(self.num_experts, 256), len(source.segments))](
                buffers.step_map, self.segment_offsets[layer_index],
                self.pointer_tables[layer_index], bank.data_ptr(),
                ROW_BYTES=bank.shape[1] * 4, EXPERTS=self.num_experts, BLOCK=256,
            )
        else:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Shared EXL3 prefill staging is eager-only")
            self.prefill_bank[:, :source.data.shape[1]].copy_(source.data, non_blocking=True)
            state = self.prefill_states[layer_index]
        out = apply_exl3_fused_moe(
            x, ids, weights, state, state._exl3_inners, None, limit
        )
        return out.to(x.dtype)

    def snapshot(self):
        """Read back cache invariants and footprint outside the forward path."""
        for table in self.policy_tables.values():
            policy.check_global_tables(table)
        resident = [int((table.layer_slice(table.hot_phys, i) >= 0).sum())
                    for table, i in self.layer_tables]
        result = {
            "cache_policy": self.cache_policy,
            "resident_per_layer": resident,
            "bank_bytes": sum(b.numel() * b.element_size() for b in self.banks.values()),
            "prefill_bytes": self.prefill_bank.numel() * self.prefill_bank.element_size(),
            "host_bytes": sum(s.data.numel() * s.data.element_size() for s in self.sources),
            "last_copies": [int(b.gather_count[0]) for b in self.buffers],
            "slot_bytes": self.row_words * 4,
        }
        return result
