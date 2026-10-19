# MGAHAE

This package contains the core implementation and datasets for multi-view heterogeneous graph community detection. It combines an original heterogeneous graph, a semantic homogeneous graph constructed from dataset-specific meta-paths, and an attribute similarity graph.

The experiment settings in Sections 2–4 were supplied by the author, including the candidate sets and final configurations transcribed from the supplied manuscript screenshots. Section 5 documents the behavior of the source code currently included in this package. The declared protocol and the legacy implementation are not yet fully aligned; the cleanup does not constitute a new experiment or a verified reproduction.

## 1. Package contents

```text
MGAHAE/
├── main.py                 # Training entry point and evaluation calls
├── data.py                 # Dataset loading, graph construction, configuration
├── model.py                # Multi-view model and reconstruction decoders
├── model_han_gl.py         # Type-specific projections and graph-view assembly
├── model_oslayer.py        # GAT, GCN, and semantic attention layers
├── evaluate.py             # Community detection evaluation
├── utils.py                # Early stopping and legacy helper functions
├── README_MGAHAE.md
└── Dataset/
    ├── ACM/
    ├── DBLP/
    └── YELP/
```

The `Dataset` directory contains only the three datasets used in Table 3: DBLP, ACM3025, and Yelp. Their 29 existing data files include features, graph relations, meta-path matrices, labels, and existing index files, and were preserved without modification. A dataset consists of multiple required files, rather than a single file. `ACM` in the supplied Table 3 corresponds to the four-meta-path `ACM3025` variant now stored in `Dataset/ACM` (formerly `Dataset/MGAG_ACM3025`). The original ACM4019 dataset and the `ie_ACM4025` and `ie_IMDB3228` directories were archived outside the package; the current `ACM` directory contains ACM3025, not ACM4019.

Historical experiment results, model weights, TensorBoard files, plotting scripts, the reference PDF, IDE settings, caches, an unused early-stopping module, and the old raw-data directory have been removed from this distribution directory. The active entry point uses `data.load_ourdata()` and `utils.EarlyStopping`; the old `utils.load_data()` helpers are outside the supported entry-point path and refer to the archived raw-data directory.

## 2. Experimental environment and dependencies

The author-specified experimental environment is:

| Item | Setting |
|---|---|
| Deep learning framework | PyTorch 2.9.1 |
| CUDA | 12.8 |
| GPU | One NVIDIA GeForce RTX 5090 |
| GPU memory | 32 GB |

The source imports the following third-party packages:

```text
torch
dgl
numpy
scipy
scikit-learn
networkx
python-louvain
munkres
tensorboard
tqdm
```

`python-louvain` provides the `community` module. Python, DGL, and the other dependency versions were not supplied and are not invented here. Compatibility of this legacy source with PyTorch 2.9.1 / CUDA 12.8 has not been tested during packaging.

## 3. Author-specified parameter settings

### Graph construction and architecture

Dataset-specific meta-paths construct semantic homogeneous graphs. Attribute similarity graphs use heat-kernel-based similarity and the K-nearest neighbor (KNN) strategy. The number of nearest neighbors is fixed at `k = 300`, and the heat-kernel bandwidth is fixed at `t = 1`. These graph construction settings are predefined according to dataset characteristics and do not use ground-truth community labels.

A single-layer graph attention network (GAT) encodes the original heterogeneous graph. Two single-layer graph convolutional networks (GCNs) encode the semantic homogeneous graph and the attribute similarity graph, respectively. Type-specific features are projected into a shared `d`-dimensional space; the view encoders produce `d`-dimensional representations. The semantic attention scoring network has dimensions `d → 128 → 1`. The GAT uses one attention head and a dropout rate of `0.35`.

The number of communities `K` is fixed by the predefined community detection setting for each dataset. Here `k` denotes the KNN neighborhood size, while `K` denotes the number of communities.

### Optimization and label-free model selection

The optimizer is Adam with a weight decay of `0.001`, for a maximum of `800` epochs. Early stopping monitors a label-free validation loss with a patience of `100` epochs. The checkpoint with the lowest validation loss is retained.

The loss coefficients `α`, `β`, `γ`, and embedding dimension `d` are selected by maximizing community modularity `Q`, computed from the semantic homogeneous graph and the predicted soft community-assignment matrix `C`. Ground-truth labels are reserved for final performance evaluation and post hoc parameter sensitivity analysis. They must not influence training, hyperparameter selection, early stopping, checkpoint selection, or the choice of prediction output.

The supplied candidate sets are:

| Hyperparameter | Candidate values |
|---|---|
| `α` | `{0, 20, 40, 60, 80, 100, 120, 140, 160}` |
| `β` | `{0, 0.1, 1, 10, 30, 60, 90, 120, 150, 180}` |
| `γ` | `{0, 1e-4, 1e-3, 1e-2, 1e-1, 1}` |
| `d` | `{4, 8, 16, 32, 64, 128, 256}` |

These are the author-supplied search spaces, not evidence that a search was executed in this package. A validation split, an exact validation-loss definition, the search schedule, and the tie-breaking/seed aggregation rules were not supplied. No additional protocol choices or completed search results are asserted here.

## 4. Table 3: Author-supplied final configurations

| Dataset | Meta-paths | Learning rate | `α` | `β` | `γ` | `d` | `K` |
|---|---|---:|---:|---:|---:|---:|---:|
| DBLP | APA, APTPA, APCPA | 0.005 | 120 | 180 | 1e-4 | 128 | 4 |
| ACM | PAP, PSP, PSPAP, PAPSP | 0.01 | 60 | 120 | 1e-3 | 64 | 3 |
| Yelp | BUB, BSB, BLB, BUBSB, BUBLB | 0.01 | 80 | 0.1 | 0.1 | 8 | 3 |

The DBLP loader names its conference/venue meta-path file `apvpa.npz`; the supplied table calls that relation `APCPA`. This preserves the manuscript notation without renaming the data file.

## 5. Current source behavior and differences

This section describes the included source, rather than the intended experimental protocol above.

| Item | Included implementation |
|---|---|
| Default dataset | `data.setup()` forces `ACM3025`. |
| Dataset parameters | `data.setup()` overrides the defaults in `main.py`; they differ from Table 3. |
| Shared feature projection | `hidden_units = 64` in `main.py`. |
| View representation dimension | GAT/GCN output width is hard-coded to `32` in `model_oslayer.OSGNN`; attention is `32 → 128 → 1`. Changing `hidden_units` alone does not change that output width. |
| Attribute graph | For the three included datasets: cosine KNN with `k = 300` for DBLP and Yelp; cosine threshold `0.29` for ACM3025. No heat-kernel bandwidth is implemented. |
| Early stopping | `main.py` computes NMI using ground-truth labels every epoch and passes it to `utils.EarlyStopping.step()`. The stopping/checkpoint decisions use NMI, not a held-out label-free validation loss. |
| Checkpoint used for evaluation | The best-checkpoint reload in `main.py` is commented out; evaluation uses the final model state. |
| Hyperparameter search | No automatic candidate search or modularity-based configuration selection is implemented. |
| Final prediction choice | `evaluate.metric_sets()` compares K-means and direct-assignment NMI using labels and returns the higher-NMI result. |
| Modularity and entropy outputs | `evaluate.py` returns placeholder values `Q = 66.6` and `AE = 66.6`; these are not measured metrics. |
| Device handling | Some operations explicitly use CUDA or `cuda:0`; CPU execution is not fully supported by the current implementation. |

The source training objective is:

```text
loss = α * structure_BCE + β * attribute_BCE - γ * modularity_term
modularity_term = trace(C.T @ B @ C)
```

`main.py` stores these coefficients under the keys `"α"`, `"b"`, and `"c"`, respectively. This source modularity term is used in the training loss; it does not implement the author-specified hyperparameter selection procedure.

The current settings for the three included datasets, after `data.setup()` overrides, are:

| Code dataset name | Learning rate | `α` | `β` (`b`) | `γ` (`c`) | Maximum epochs | Patience |
|---|---:|---:|---:|---:|---:|---:|
| DBLP | 0.005 | 50 | 150 | 8e-5 | 800 | 100 |
| ACM3025 (default) | 0.01 | 10 | 5 | 1e-5 | 800 | 10 |
| yelp | 0.01 | 0 | 0 | 1 | 3 | 100 |

Legacy loaders and configuration branches for ACM4019, ACM4025, and IMDB remain in the source, but their datasets are not included in this distribution. They are not used by the default entry point. In particular, the legacy `load_ACM_data()` expects ACM4019 files and must not be used for the current `Dataset/ACM` directory; that directory is read by `load_ourACM3025()`. The three included datasets map to the dispatcher names `DBLP`, `ACM3025`, and `yelp`, respectively.

## 6. Running the included entry point

From PowerShell, with the required dependencies already installed:

```powershell
Set-Location 'C:\Users\admin\Desktop\MGAHAE'
python main.py
```

This invokes the current legacy ACM3025 configuration documented in Section 5. It does not run the Table 3 protocol. The current CLI exposes `--log-dir`; dataset and training settings are configured inside the source rather than through additional CLI flags.

New runs create `results/.../result.txt` and TensorBoard event files through `SummaryWriter`, and save checkpoints under `parameters/`. These runtime outputs are absent from the cleaned distribution. `utils.EarlyStopping.save_checkpoint()` now creates its output directory automatically, so removing historical checkpoint directories does not cause a missing-directory failure.

Packaging verification checks syntax, local module dependencies, retained-file hashes, and the availability of files read by the active dataset loaders. No training run, CUDA compatibility test, hyperparameter search, or new experimental metric was produced during this cleanup.
