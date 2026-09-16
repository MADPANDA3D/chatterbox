# Experimental Turbo alignment capture

`TurboAlignmentCapture` observes the existing GPT-2 Q/K projections during one
Turbo generation. It does not switch attention kernels, request full attention
matrices, modify logits, or run another neural model. It installs temporary
hooks only on explicitly selected layers and removes them on normal or failed
exit. Normal `generate()` behavior and return values remain unchanged.

```sh
python example_turbo_alignment.py --text 'First came a boom, then another boom, and finally silence.' --audio-prompt /path/to/reference.wav --output /tmp/turbo-trace
python -m unittest discover -s tests -p test_turbo_alignment.py
```

The example's heads `(4, 6)`, `(5, 14)`, `(8, 4)` showed ordered text progression
in a small Turbo exploration. They are diagnostic examples, not a calibrated
or universal head selection. Nano and multilingual models were not validated.

## Output and limits

- The tensor has axes `[selected head, decoding step, normalized text token]`.
  Subword pieces and repeated words retain distinct token positions.
- Softmax is over the text keys only. Scores are **not** probabilities over the
  full context and cannot establish how much attention was paid to text versus
  the conditioning/audio context.
- The first row is the speech-BOS query used to predict the first speech token;
  subsequent rows observe successive decode queries. An EOS prediction may be
  present. Existing generation then filters invalid tokens and adds trailing
  silence before converting speech tokens to a waveform. These rows are not
  automatically waveform positions.
- Word-boundary accuracy, multilingual behavior, paralinguistic tags, long
  inputs, and failure/confidence thresholds remain unvalidated. In particular,
  do not use an attention argmax as a guaranteed audible word-end timestamp.
- Use one utterance and one active generation on the model. Match the exact
  normalized/truncated text-token count; the API does not infer it. Capture
  memory grows with the selected heads and decoding steps. `max_steps` bounds
  recording and raises on overflow; it defaults to the current Turbo limit plus
  its prefill prediction. Read the trace after leaving the context.

## Validation

CPU tests use a tiny randomly initialized GPT-2: exact hidden-output parity,
projected-QK numerical agreement, unchanged attention configuration, invalid
arguments, bounded recording, cleanup after exceptions and recorder reuse.
They require no checkpoint downloads.

A separate offline GPU check used the existing cached Turbo checkpoint on an
RTX 3060, PyTorch 2.6.0+cu124 and Transformers 5.2.0. After one warmup, three
seed-matched baseline/capture pairs produced **bit-identical waveform tensors**
(maximum absolute difference zero) and left zero hooks installed:

| Input | Audio seconds | Baseline seconds | Capture + extraction seconds |
| --- | ---: | ---: | ---: |
| First came a boom, then another boom, and finally silence. | 4.28 | 1.622 | 1.614 |
| The red boat passed the blue boat before sunset. | 2.96 | 1.168 | 1.169 |
| Hello! Really? Yes, really. | 2.32 | 0.945 | 0.935 |

These single short runs establish feasibility, not a speedup, statistical
performance bound, general output-parity guarantee or word-timing accuracy.
No reference voice, generated audio, checkpoint, credential or machine-specific
configuration is distributed with this change. No live TTS service was changed.
