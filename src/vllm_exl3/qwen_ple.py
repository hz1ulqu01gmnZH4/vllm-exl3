"""Adapter for the current Qwen4Exp PLE module interface (single GPU)."""

from torch import nn

from .exl3 import Exl3EmbeddingMethod


class Exl3QwenPLEEmbedding(nn.Module):
    supports_prefetch = False

    def __init__(
        self, num_embeddings, embedding_dim, *, quant_config, prefix, params_dtype
    ):
        super().__init__()
        from vllm.distributed import get_etp_group, get_tp_group

        if get_etp_group().world_size != 1 or get_tp_group().world_size != 1:
            raise ValueError("EXL3 Qwen PLE currently requires TP=ETP=1")
        spec = quant_config._ngram_embedding_spec(prefix)
        if spec is None:
            raise ValueError(f"Missing EXL3 PLE table specification: {prefix}")
        self.tp_rank, self.tp_size = 0, 1
        self.quant_method = Exl3EmbeddingMethod(quant_config, spec)
        self.quant_method.create_weights(
            self,
            embedding_dim,
            [num_embeddings],
            embedding_dim,
            num_embeddings,
            params_dtype,
        )

    @property
    def weight(self):
        return self.shard_0.trellis if self.quant_method.sharded else self.trellis

    def forward(self, input_ids):
        return self.quant_method.embedding(self, input_ids)

    def dequantize(self, embeddings, output_dtype):
        return embeddings.to(output_dtype)
