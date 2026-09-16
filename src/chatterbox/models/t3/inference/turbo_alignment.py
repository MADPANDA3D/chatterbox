"""Opt-in GPT-2 text/speech alignment diagnostics, without changing inference.

These are text-normalized QK scores, not calibrated word timestamps. Different
heads may lead or lag speech. Keep the usual SDPA/Flash attention implementation;
only the already-computed projections are observed.
"""

import math

import torch


class TurboAlignmentCapture:
    """Capture one Turbo generation with explicit (layer, head) selections.

    ``text_token_count`` must match the normalized/truncated input passed to T3.
    Rows correspond to decoding steps (including an EOS prediction, if emitted),
    columns to input text tokens. Conditioning and the initial speech BOS precede
    generation. No inference, sampling, logits or attention configuration changes.
    The same model must not be used concurrently while this context is active.
    """

    def __init__(self, transformer, text_token_count, heads, max_steps=1001):
        self.transformer = transformer
        self.text_token_count = text_token_count
        self.heads = tuple(heads)
        self.max_steps = max_steps
        if not isinstance(text_token_count, int) or text_token_count < 1:
            raise ValueError("text_token_count must be a positive integer")
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not self.heads or len(set(self.heads)) != len(self.heads):
            raise ValueError("Provide distinct (layer, head) pairs")
        for layer, head in self.heads:
            if not (0 <= layer < len(transformer.h) and 0 <= head < transformer.config.n_head):
                raise ValueError("Alignment head is outside this GPT-2 model")
        self._handles = []
        self._keys = {}
        self._queries = {}

    def __enter__(self):
        if self._handles:
            raise RuntimeError("Alignment capture is already active")
        self._keys.clear()
        self._queries.clear()
        try:
            for layer in dict.fromkeys(layer for layer, _ in self.heads):
                module = self.transformer.h[layer].attn.c_attn
                self._handles.append(module.register_forward_hook(self._hook(layer)))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook(self, layer):
        selected = [head for selected_layer, head in self.heads if selected_layer == layer]

        def observe(module, args, output):
            if output.shape[0] != 1:
                raise ValueError("Alignment capture requires single-utterance inference")
            width = output.shape[-1] // 3
            query, key, _ = output.split(width, dim=-1)
            n_heads = self.transformer.config.n_head
            head_dim = width // n_heads
            if layer not in self._keys:
                start = output.shape[1] - self.text_token_count - 1
                if start < 0:
                    raise ValueError("First capture must include conditioning, text and speech BOS")
                text_keys = key[0, start:start + self.text_token_count]
                self._keys[layer] = text_keys.reshape(-1, n_heads, head_dim).transpose(0, 1)[selected].detach().clone()
                self._queries[layer] = []
            if len(self._queries[layer]) >= self.max_steps:
                raise ValueError("Alignment capture exceeded max_steps")
            self._queries[layer].append(query[0, -1].reshape(n_heads, head_dim)[selected].detach().clone())
            # No return value: PyTorch leaves the model's original output intact.

        return observe

    def text_attention(self):
        """Return CPU float32 [selected_heads, decoding_steps, text_tokens].

        Softmax is over text positions only, not the full model context. This
        diagnostic cannot establish how much attention the model paid to text
        versus conditioning/audio, or guarantee audible word boundaries.
        """
        if self._handles:
            raise RuntimeError("Read alignment after generation leaves the context")
        if not self._queries:
            raise RuntimeError("No generation was captured")
        if (set(self._queries) != {layer for layer, _ in self.heads}
                or len({len(rows) for rows in self._queries.values()}) != 1):
            raise RuntimeError("Generation ended with an incomplete attention step")
        by_layer = {}
        for layer, rows in self._queries.items():
            query = torch.stack(rows, dim=1).float()
            keys = self._keys[layer].float()
            scores = query @ keys.transpose(-1, -2) / math.sqrt(keys.shape[-1])
            attention = self.transformer.h[layer].attn
            if not attention.scale_attn_weights:
                scores *= math.sqrt(keys.shape[-1])
            if attention.scale_attn_by_inverse_layer_idx:
                scores /= layer + 1
            by_layer[layer] = scores.softmax(-1).cpu()
        offsets = {}
        result = []
        for layer, _ in self.heads:
            offset = offsets.get(layer, 0)
            result.append(by_layer[layer][offset])
            offsets[layer] = offset + 1
        return torch.stack(result)
