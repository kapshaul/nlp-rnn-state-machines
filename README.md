# Finite-State Machines and RNNs

Historical coursework code on how recurrent networks (LSTMs) handle
finite-state behaviour: a hand-configured scalar LSTM for parity, a trained
PyTorch LSTM for parity with a length-generalization sweep, and a
bidirectional LSTM part-of-speech (POS) tagger on the TorchText UDPOS dataset.

Write-up: <https://kapshaul.github.io/studies/finite-state-machine/>

## Files

| File | Description |
| --- | --- |
| [`univariate_tester.py`](univariate_tester.py) | NumPy only. Scalar LSTM with hand-set gate weights; checks every length-14 binary string for correct parity and reports the first failure. No training. |
| [`driver_parity.py`](driver_parity.py) | PyTorch. Trains `ParityLSTM` on all binary strings up to length 5, then evaluates random strings of length 1–256 and saves `<model>_parity_generalization.png` in the working directory. |
| [`driver_udpos.py`](driver_udpos.py) | PyTorch/TorchText. BiLSTM POS tagger (`MODE = 'Train'` by default, or `'Test'`). |
| [`test_udpos.py`](test_udpos.py) | UDPOS data inspection: prints one tagged sentence and shows a POS tag histogram. It is not a unit-test suite and does not evaluate the checkpoint. |
| [`Vocabulary.pkl`](Vocabulary.pkl) | Historical pickled `Vocabulary` object from a past POS training run. |
| [`bilstm_pos_model.pth`](bilstm_pos_model.pth) | Historical BiLSTM POS `state_dict` from a past training run. |
| [`Figures/`](Figures/) | 11 historical plots: parity generalization for LSTM-1 to LSTM-256, `Loss.png`, `Histogram.png`. |

## Environment

The scripts import `numpy`, `torch`, `torchtext`, `matplotlib`, `tqdm` and
`nltk`. No versions are pinned in this repository and none are specified here.
Notes:

- `driver_udpos.py` uses `match`/`case`, so it needs Python 3.10+.
- The UDPOS scripts rely on an older TorchText API (`torchtext.datasets.UDPOS(split=...)`),
  and `test_udpos.py` also imports `torch.utils.data.backward_compatibility`.
  A historically compatible torch/torchtext pair is required; recent releases
  may not provide these APIs.
- `Test` mode's `tag_sentence` uses `nltk.word_tokenize`, which may require the
  NLTK tokenizer data to be available locally.

During the history cleanup, the NumPy scalar parity checker passed all 16,384
length-14 binary inputs. The training and POS scripts were not run.

## Usage

Run from the repository root. Each command is independent; they are not meant
to be run as a sequence.

```bash
python univariate_tester.py   # no training, no downloads
python driver_parity.py       # trains (2000 epochs) and writes a PNG
python driver_udpos.py        # downloads UDPOS; trains by default
python test_udpos.py          # downloads UDPOS; inspection only
```

- **`driver_parity.py`** trains from scratch and writes
  `LSTM-<hidden_dim>_parity_generalization.png` to the repo root (default
  `hidden_dim=1`). It does not write into `Figures/`.
- **`driver_udpos.py`, `MODE = 'Train'`** downloads UDPOS via `data_loader()`,
  **overwrites `Vocabulary.pkl`**, trains for 50 epochs, **overwrites
  `bilstm_pos_model.pth`** whenever validation loss improves, and shows a loss plot.
  Back up the committed artifacts first if you want to keep them.
- **`driver_udpos.py`, `MODE = 'Test'`** (edit the constant in the script)
  loads the preserved `Vocabulary.pkl` and `bilstm_pos_model.pth`, reports test
  accuracy and tags three example sentences. It still calls `data_loader()`, so
  UDPOS is still downloaded/loaded.
- **`test_udpos.py`** downloads UDPOS at import time and only visualizes data.

## Historical assets

[`Figures/`](Figures/), [`Vocabulary.pkl`](Vocabulary.pkl) and
[`bilstm_pos_model.pth`](bilstm_pos_model.pth) are kept as historical outputs
of earlier runs. They were not regenerated for this README, and they are not
presented as validated scientific results. See the
[write-up](https://kapshaul.github.io/studies/finite-state-machine/) for discussion.
