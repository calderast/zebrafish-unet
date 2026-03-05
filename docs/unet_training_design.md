# U-Net Training on C. elegans Nuclei: Dataset & Design Document

## Table of Contents

1. [Project Context](#1-project-context)
2. [The C. elegans Nuclei Dataset](#2-the-c-elegans-nuclei-dataset)
3. [Why This Dataset](#3-why-this-dataset)
4. [Background: Watershed Segmentation](#4-background-watershed-segmentation)
5. [Background: The U-Net Architecture](#5-background-the-u-net-architecture)
6. [Design Choices](#6-design-choices)
   - [2D vs 3D](#2d-vs-3d)
   - [Slice Extraction Strategy](#slice-extraction-strategy)
   - [Input Size (144x144)](#input-size-144x144)
   - [Target Formulation (3-Class)](#target-formulation-3-class)
   - [Automatic Boundary Generation](#automatic-boundary-generation)
   - [Model Architecture](#model-architecture)
   - [Loss Function](#loss-function)
   - [Data Augmentation](#data-augmentation)
   - [Normalization](#normalization)
   - [Training Regime](#training-regime)
   - [Post-Processing: Semantic to Instances](#post-processing-semantic-to-instances)
   - [Evaluation Metrics](#evaluation-metrics)
7. [References](#7-references)

---

## 1. Project Context

The broader goal is to train a U-Net for automated single-cell segmentation of zebrafish
embryos at different developmental stages, using Zebrahub light sheet microscopy data
(Histone-mCherry nuclear marker). The ultimate aim is to extract morphological features
per cell type and apply mechanistic interpretability to the U-Net.

The zebrafish data presents challenges: no pre-computed ground truth exists, and our
attempts to generate labels via Ultrack (notebook 03) produced very sparse masks (1-170
instances per timepoint, <3% foreground). Classical watershed (notebook 04) may improve
this but is untested and has no human validation.

Rather than train on unreliable labels, we first build and validate the full training
pipeline on an established, expert-annotated dataset — the C. elegans 3D nuclei dataset —
then transfer the model and pipeline to zebrafish data.

---

## 2. The C. elegans Nuclei Dataset

**Source:** [Zenodo record 5942575](https://zenodo.org/records/5942575)

### Origin and Curation

The raw confocal microscopy data was originally acquired by Long et al. (2009) for their
3D digital atlas of *C. elegans*. The volumes capture L1-stage (first larval stage) worms
using a Leica confocal microscope with a 63x oil-immersion objective. Each volume shows
fluorescently labeled nuclei throughout the entire worm body.

The instance segmentation masks were manually curated by Dagmar Kainmueller (MDC Berlin),
providing a unique integer label for every individual nucleus in every volume.

The train/validation/test split was established by Weigert et al. (2020) for benchmarking
their StarDist 3D method, making it a standard evaluation dataset for 3D nuclear
segmentation.

### Data Specifications

| Property | Value |
|---|---|
| Organism | *Caenorhabditis elegans*, L1 larval stage |
| Imaging | Leica confocal microscope, 63x oil objective |
| Volumes | 28 total |
| Dimensions (Y, X, Z) | 140 x 140 x 994-1275 pixels |
| Pixel size (Y, X, Z) | 0.116 x 0.116 x 0.122 µm (nearly isotropic) |
| Physical size | ~16 x 16 x 120-155 µm per volume |
| Image format | TIFF, uint8, single-channel fluorescence |
| Mask format | TIFF, uint32, instance labels (0 = background) |
| Nuclei per volume | 512-566 (mean ~541) |
| Total nuclei | ~15,000 across all volumes |
| Download size | 84.4 MB (zipped) |
| Uncompressed size | ~3.2 GB |

### Train/Val/Test Split

| Split | Volumes | Total Z-slices | Nuclei per volume |
|---|---|---|---|
| Train | 18 | ~21,000 | 512-566 |
| Validation | 3 | ~3,400 | 526-539 |
| Test | 7 | ~7,800 | 524-547 |

### Data Structure

```
c_elegans_nuclei/
├── readme.md
├── train/
│   ├── images/    (18 .tif files, uint8)
│   └── masks/     (18 .tif files, uint32)
├── val/
│   ├── images/    (3 .tif files)
│   └── masks/     (3 .tif files)
└── test/
    ├── images/    (7 .tif files)
    └── masks/     (7 .tif files)
```

Each image file and its corresponding mask file share the same filename. The volumes are
3D TIFF stacks with shape (Y, X, Z) = (140, 140, ~1050). The Z axis runs along the
anterior-posterior body axis of the worm.

### Per-Slice Statistics

When extracting 2D XY cross-sections for training:

| Property | Value |
|---|---|
| Nuclei per XY slice | median 5, mean 6.9, max ~26 |
| 2D nucleus area | median ~69 px, range ~9-139 px (p5-p95) |
| 2D equivalent diameter | ~9.4 px median (~1.1 µm) |
| Foreground fraction | mean 2.5%, max ~9.7% |
| Empty slices (0 nuclei) | ~1.7% |
| Slices with ≥3 nuclei | ~79% |
| Touching nuclei pairs | ~11 out of 16 at a dense slice |

### Volume Naming Convention

Filenames encode the transgenic reporter line and acquisition metadata. Examples:

- `C18G1_2L1_1.tif` — C18G1.2 reporter, L1 stage
- `cnd1threeL1_1213061.tif` — cnd-1 reporter (third construct), L1
- `hlh1fourL1_0417075.tif` — hlh-1 reporter (fourth construct), L1
- `pha4I2L_0408072.tif` — pha-4 reporter, L1
- `mir61L1_1229062.tif` — mir-61 reporter, L1

Different reporter lines produce different fluorescence patterns (some genes are expressed
in specific tissues), but the nuclear staining and morphology are consistent across all
volumes. This variation is beneficial — the model must learn to segment nuclei regardless
of which tissue has brighter reporter expression.

---

## 3. Why This Dataset

### The task is nearly identical

We want to segment fluorescent nuclei in zebrafish light sheet microscopy. This dataset is
fluorescent nuclei in C. elegans confocal microscopy. The visual problem is the same:
bright blobs on a dark background, varying sizes, touching and overlapping instances. A
model that learns "what a fluorescent nucleus looks like" here will transfer meaningfully
to zebrafish.

### The labels are high quality

This is manually curated, expert-annotated dense instance segmentation. Every nucleus in
every volume has a unique ID. Compare this to our zebrafish labels:

- **Ultrack (notebook 03):** Found only 1-170 instances per timepoint, extremely sparse,
  <3% foreground. The default preprocessing was too conservative for the low-contrast
  zebrafish data.
- **Watershed (notebook 04):** Untested, fully automated, no human validation.

Training on noisy or incomplete labels teaches the model to reproduce the noise. Training
on clean, expert-verified labels teaches it actual nuclear morphology.

### It's the right size for prototyping

28 volumes, 84 MB compressed. Fits entirely in memory (~3 GB uncompressed), trains in
minutes on CPU. Large enough to learn real features, small enough to iterate fast. We can
debug the full pipeline — dataset class, augmentation, model, loss, training loop,
post-processing, evaluation — before spending hours processing zebrafish data.

### It has a published benchmark split

The 18/3/7 train/val/test split was established by Weigert et al. (2020) for StarDist 3D
benchmarking. This means we can compare our U-Net's performance against published results
on the exact same data.

### The nuclei are densely packed

~540 nuclei in a 140x140 cross-section means nuclei frequently touch with no gap between
them. This is the hard case for instance segmentation and the reason the 3-class
(background/boundary/interior) formulation matters. If the model can separate touching
C. elegans nuclei, it can handle touching zebrafish nuclei.

### What it doesn't give us

The imaging modality differs (confocal vs light sheet), the organism differs (worm vs
fish), the scale differs (0.116 µm/px vs 0.439 µm/px), and nuclear morphology differs
somewhat. A model trained purely on this data won't be perfect on zebrafish, but it gives
us a validated pipeline and a pre-trained starting point for fine-tuning.

---

## 4. Background: Watershed Segmentation

Imagine pouring water onto a topographic surface. The water pools in valleys and rises
until pools from different valleys meet at ridgelines. Those ridgelines become the
boundaries between regions.

In image segmentation, the "topography" is typically the **negative distance transform**
of a binary mask. Each foreground pixel gets a value equal to its distance to the nearest
background pixel. The centers of objects become "valleys" (most negative = farthest from
edges), and the spaces between objects become "ridgelines."

### The Algorithm

1. **Start with a binary mask** (foreground vs background), typically obtained by
   thresholding the image
2. **Compute the distance transform** — each foreground pixel gets its Euclidean distance
   to the nearest background pixel
3. **Find local maxima** of the distance transform — these are the "seeds," one per object,
   located at each object's center
4. **Flood from each seed** outward, assigning pixels to the nearest seed. Where two floods
   meet becomes the boundary between instances

```
Binary mask          Distance transform       Watershed result
┌───────────────┐    ┌───────────────┐        ┌───────────────┐
│  ███████████  │    │  1 2 3 2 1 0  │        │  AAAAAA|BBBBB │
│  ██████  ███  │    │  1 2 3 2 1    │        │  AAAAAA|BBBBB │
│  ███████████  │    │  1 2 3 2 1 0  │        │  AAAAAA|BBBBB │
└───────────────┘    └───────────────┘        └───────────────┘
Two touching blobs    Centers have highest    Split at the ridgeline
                      distance values         where floods meet
```

### In This Project

Watershed is used in two places:

- **Notebook 04 (ground truth generation):** Watershed is the *primary* segmentation
  method. Otsu thresholding → distance transform → peak detection → watershed. This
  produces instance labels from raw images.

- **Notebook 06 (U-Net post-processing):** Watershed is the *post-processing* step after
  the U-Net predicts 3-class semantic output. Connected components of the "interior" class
  become seeds, and watershed expands them into the "boundary" pixels to recover full
  nucleus areas.

---

## 5. Background: The U-Net Architecture

The U-Net was introduced by Ronneberger, Fischer, and Brox (2015) for biomedical image
segmentation, specifically for segmenting cells in microscopy images.

### The Problem It Solves

Before U-Net, convolutional networks for segmentation (like FCN) faced a fundamental
tension: you need downsampling (pooling) to capture large-scale context ("is this a cell
or background?"), but downsampling destroys spatial detail ("where exactly is the cell
boundary?"). You can't have both.

### The Architecture

The U-Net has an encoder-decoder structure shaped like the letter U:

```
Input ──► [Enc1] ─────────────────────── skip ──► [Dec1] ──► Output
              │                                      ▲
              ▼ pool                            up   │
          [Enc2] ─────────────────── skip ──► [Dec2]
              │                                  ▲
              ▼ pool                        up   │
          [Enc3] ──────────── skip ──► [Dec3]
              │                          ▲
              ▼ pool                up   │
              [Bottleneck] ──────────┘

Encoder: compress spatially, increase channels (captures "what")
Decoder: expand spatially, decrease channels (recovers "where")
Skip connections: preserve fine spatial detail across the U
```

**Encoder (left side):** Each level applies two 3x3 convolutions (with BatchNorm and ReLU),
then a 2x2 max-pooling that halves the spatial dimensions and doubles the number of
channels. This progressively compresses the image into a compact, high-level
representation. By the bottleneck, the network "knows" what objects are present but has
lost precise spatial information.

**Decoder (right side):** Each level upsamples (via transposed convolution), concatenates
the corresponding encoder feature maps via the skip connection, then applies two 3x3
convolutions. This progressively recovers the spatial resolution.

**Skip connections (the key innovation):** The encoder feature maps at each level are
copied directly to the corresponding decoder level and concatenated channel-wise. This
gives the decoder access to both:

- **High-level context** from the bottleneck: "there is a nucleus here"
- **Fine spatial detail** from the encoder: "its boundary is at exactly these pixels"

Without skip connections, the decoder would have to reconstruct spatial detail from the
compressed bottleneck representation alone, producing blurry, imprecise segmentations.

### Why It Works Well for Microscopy

- Microscopy images have repetitive local structure (nuclei look similar) — the encoder
  learns this efficiently
- Precise boundaries matter for instance segmentation — skip connections preserve them
- It works with very small training sets — the original paper used only 30 images with
  heavy augmentation
- The architecture is simple and trains stably
- It naturally handles multi-scale features — large cells and small cells in the same image

### Our Implementation vs. the Original

| Property | Original (2015) | Ours |
|---|---|---|
| Input size | 572 x 572 | 144 x 144 |
| Depth (pooling levels) | 4 | 3 |
| Channel progression | 64→128→256→512→1024 | 32→64→128→256 |
| Parameters | ~31M | ~1.9M |
| Output | 2-class (cell/bg) | 3-class (bg/boundary/interior) |
| Padding | None (output smaller than input) | Same padding (output = input size) |

The same fundamental architecture, scaled down because our images are smaller and our
compute budget is smaller (CPU training).

---

## 6. Design Choices

### 2D vs 3D

**Choice: 2D (XY slices)**

The C. elegans volumes are 140x140x~1050. A 3D U-Net would process 3D patches (e.g.,
144x144x32), requiring more parameters, much slower training, and more complex data
loading. The voxel size is nearly isotropic (0.116 x 0.116 x 0.122 µm), so there's no
strong physical reason to favor 3D — a nucleus looks roughly the same in XY as in XZ.

Starting with 2D allows fast iteration on CPU, simple debugging, and establishes baseline
metrics. The architecture can be upgraded to 3D later if the 2D results are promising.

### Slice Extraction Strategy

**Choice: Every 2nd slice, minimum 3 nuclei per slice**

**Stride=2:** Adjacent z-slices are nearly identical — they're only 0.122 µm apart, while
nuclei are ~1-2 µm in diameter. A nucleus appears across ~8-16 consecutive slices with
minimal visual change between neighbors. Training on every slice means the model sees
near-duplicate data, wasting compute and risking overfitting to specific worm anatomies
rather than learning generalizable features. Stride=2 halves the dataset with minimal
information loss.

**min_nuclei=3:** Slices near the tips of the worm or between nuclei in Z can have 0-2
nuclei. These are uninformative — the model learns almost nothing from an image that is
99.9% background with one partial nucleus at the edge. Filtering to ≥3 nuclei focuses
training on slices with meaningful segmentation structure.

**Resulting dataset:** ~10,000 training slices (from 18 volumes) after stride=2 and
filtering, ~1,700 validation slices (from 3 volumes). In fast dev mode, this is further
sub-sampled to 500 training slices.

### Input Size (144x144)

**Choice: Reflect-pad from 140x140 to 144x144**

A U-Net with depth D requires input dimensions divisible by 2^D. With depth=3, we need
divisibility by 8. The raw slices are 140x140.

| Option | Size | Issue |
|---|---|---|
| Center crop | 128x128 | Loses ~16% of area. Nuclei at edges get cut off. |
| Use as-is | 140x140 | 140→70→35 (odd). Shape mismatches in skip connections. |
| **Reflect pad** | **144x144** | **144→72→36→18. All clean. Only 2px padding per side.** |
| Zero pad | 160x160 | Works, but 10px of artificial black border. Wasteful. |

Reflect padding mirrors the image content at the boundary, avoiding artificial edges that
zero padding would create. This matters because nuclei appear right at the image edges.

### Target Formulation (3-Class)

**Choice: Background (0) / Boundary (1) / Interior (2)**

This is the most important design choice. Consider what happens with binary (foreground vs
background): when two nuclei touch — which happens frequently (~11 out of 16 nuclei touch
neighbors in dense slices) — binary segmentation produces a single merged blob.
Post-processing with watershed can try to split them, but without explicit boundary
information it often fails or splits incorrectly.

The 3-class approach teaches the model to predict a **boundary class** at the contact
zone between nuclei. These boundary pixels act as "fences" that separate instances.
Post-processing then becomes simple: connected components of the interior class
automatically yields one component per nucleus.

```
Binary (2-class):                   3-class:

  ██████████████                    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  ██████████████  ← merged!        ██████▓▓████████  ← separated!
  ██████████████                    ██████▓▓████████
  ██████████████                    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓

  █ = foreground                   █ = interior (class 2)
                                   ▓ = boundary (class 1)
```

**Why not distance transform regression?** Predicting the distance from each pixel to the
nearest boundary is another valid approach, but it requires a regression loss, the
distances are small (nuclei are ~9px diameter, max distance ~4-5px), and it's harder to
tune. 3-class semantic segmentation is simpler and well-proven.

This approach comes from the original U-Net paper (Ronneberger et al. 2015, which used
weighted loss at boundaries) and is also used by StarDist and other nuclear segmentation
methods.

### Automatic Boundary Generation

The boundary class labels are generated automatically from the instance masks — no manual
boundary annotation is needed.

```python
def instance_to_semantic(mask_slice):
    target = np.zeros_like(mask_slice, dtype=np.uint8)
    target[mask_slice > 0] = 2  # all foreground → interior
    bnd = find_boundaries(mask_slice, mode="inner")
    target[bnd] = 1  # overwrite edges → boundary
    return target
```

`find_boundaries` (from scikit-image) examines each pixel's neighbors. If any neighbor has
a **different label**, that pixel is marked as a boundary. `mode="inner"` places boundary
pixels inside each nucleus (the outermost ring of pixels), rather than in the background
between them.

This means:
- Where two differently-labeled nuclei are adjacent, both get their touching edges marked
  as boundary
- Where a nucleus borders background, its outer ring is also marked as boundary
- The interior class remains a clean, connected region for each nucleus

### Model Architecture

**Choice: Depth-3 U-Net, channels [32, 64, 128, 256], ~1.9M parameters**

**Depth-3** (3 max-pooling levels): The input is 144x144. Each pooling halves spatial
dimensions: 144→72→36→18 (bottleneck). With depth-4, the bottleneck would be 9x9 — too
small to capture meaningful spatial patterns. Depth-3 gives an 18x18 bottleneck, which is
reasonable for our image content.

**Channel progression [32, 64, 128, 256]:** Standard U-Net pattern — double channels at
each level as spatial resolution halves. This keeps computational cost roughly balanced
across levels. Starting at 32 (not 64 like the original U-Net) keeps the model small for
fast CPU training.

**~1.9M parameters:** Small enough to train on CPU in minutes, large enough to learn the
3-class task on 140x140 images. The original U-Net had ~31M parameters for 572x572 images;
our smaller inputs and simpler structure don't need that capacity.

#### Layer Details

Each encoder/decoder block is a "double conv": Conv3x3 → BatchNorm → ReLU → Conv3x3 →
BatchNorm → ReLU.

- **BatchNorm** stabilizes training by normalizing activations between layers, allowing
  higher learning rates and faster convergence. Without it, learning rate tuning becomes
  much more sensitive. The Conv2d layers use `bias=False` because BatchNorm already has a
  learnable bias (its β parameter).

- **ConvTranspose2d** for upsampling (learned upsampling in a single operation) rather than
  bilinear interpolation + convolution. Simpler (one layer instead of two) and matches the
  original U-Net design. Checkerboard artifacts are not a concern at this scale.

#### Spatial Dimension Trace

```
Input:       [B,   1, 144, 144]
Enc1:        [B,  32, 144, 144]  → pool → [B,  32,  72,  72]
Enc2:        [B,  64,  72,  72]  → pool → [B,  64,  36,  36]
Enc3:        [B, 128,  36,  36]  → pool → [B, 128,  18,  18]
Bottleneck:  [B, 256,  18,  18]
Dec3: up →   [B, 128,  36,  36]  cat enc3 → [B, 256,  36,  36] → conv → [B, 128, 36, 36]
Dec2: up →   [B,  64,  72,  72]  cat enc2 → [B, 128,  72,  72] → conv → [B,  64, 72, 72]
Dec1: up →   [B,  32, 144, 144]  cat enc1 → [B,  64, 144, 144] → conv → [B,  32, 144, 144]
Output:      [B,   3, 144, 144]  (1x1 conv, no activation — raw logits)
```

### Loss Function

**Choice: 50% Weighted Cross-Entropy + 50% Dice Loss**

The class distribution is extremely imbalanced:

| Class | Fraction | Ratio to boundary |
|---|---|---|
| Background | 97.7% | 119x |
| Boundary | 0.82% | 1x |
| Interior | 1.48% | 1.8x |

With standard cross-entropy, the model can achieve 97.7% accuracy by predicting
all-background. It would never learn to detect nuclei.

**Weighted cross-entropy** with weights [1.0, 10.0, 8.0] penalizes boundary and interior
misclassifications 10x and 8x more than background errors. The weights are roughly
sqrt-inverse-frequency: boundary is ~119x rarer than background, but using weight 119
would destabilize training. Empirical weights of 5-15 work well.

**Dice loss** measures overlap rather than per-pixel accuracy:

```
Dice = 2 × |Prediction ∩ Ground Truth| / (|Prediction| + |Ground Truth|)
```

It's naturally robust to class imbalance because it cares about the *ratio* of overlap to
total area, not absolute pixel counts. Even if a class has only 100 pixels, Dice gives
meaningful gradient signal.

**Why combine both?** Cross-entropy provides stable per-pixel gradients early in training
(when the model is random and predictions are garbage). Dice provides overlap-aware
gradients later (good for refining boundaries). The 50/50 combination gets the benefits of
both. This is standard practice in medical image segmentation — nnU-Net, the most
successful general-purpose medical segmentation framework, uses exactly this combination.

### Data Augmentation

**Included:**

| Augmentation | Probability | Details | Rationale |
|---|---|---|---|
| Horizontal flip | 50% | `np.flip(axis=1)` | Free 2x data multiplier |
| Vertical flip | 50% | `np.flip(axis=0)` | Nuclei are rotationally symmetric |
| 90° rotation | 50% | `np.rot90(k={1,2,3})` | Completes 8-fold dihedral group |
| Intensity jitter | 50% | ×[0.8, 1.2] + [-0.1, 0.1] | Simulates brightness variation |
| Gaussian noise | 30% | N(0, 0.02) | Simulates sensor noise |

The flips and rotations form the **8-fold dihedral symmetry group** — the set of all
transformations that map a square onto itself. For isotropic 2D data where nuclei look the
same from all orientations, these effectively multiply the training set by 8x at near-zero
computational cost. They are the highest-impact augmentations for this data type.

Intensity jitter addresses the considerable brightness variation across volumes (intensity
maxima range from 128 to 253 across the 28 volumes). The model must be robust to this.

Spatial augmentations (flips, rotations) are applied identically to both image and mask.
Intensity augmentations are applied to the image only.

**Not included (and why):**

| Augmentation | Why excluded |
|---|---|
| Elastic deformation | Expensive on CPU; nuclei are rigid blobs, not deformable tissue |
| Arbitrary rotation | Requires interpolation that blurs the small 140x140 images |
| Scaling | Same interpolation blurring concern |
| Cutout/erasing | Adds complexity without clear benefit for this task |
| Color augmentation | Single-channel grayscale — not applicable |

### Normalization

**Choice: Per-slice percentile normalization (p1, p99) to [0, 1]**

```python
def normalize_slice(img_slice):
    p1, p99 = np.percentile(img_slice, [1, 99])
    return np.clip((img_slice - p1) / (p99 - p1), 0, 1)
```

Raw images are uint8 with highly variable intensity distributions: one volume's max is 128,
another's is 253. Simple min-max normalization (divide by 255) would not account for this
variation. Min-max per slice would be dominated by outlier bright/dark pixels.

Percentile normalization clips to [1st percentile, 99th percentile] before scaling,
making it robust to outliers while producing consistent intensity ranges across samples.
This is standard practice in microscopy image analysis.

### Training Regime

**Optimizer: Adam, lr=1e-3, weight_decay=1e-4**

Adam with lr=1e-3 is the reliable default for deep learning. It maintains per-parameter
adaptive learning rates, which helps when different layers have different gradient
magnitudes (common in U-Nets due to skip connections). Weight decay (L2 regularization)
adds a small penalty on large weights, reducing overfitting.

**Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)**

If validation loss stops improving for 5 consecutive epochs, the learning rate is halved.
This allows fine-grained learning in later training stages. Simpler than cosine annealing
and well-suited to short training runs where you don't know the optimal number of epochs
in advance.

**Batch size: 16**

For 144x144 single-channel images on CPU, batch 16 is fast (~5 MB per batch). Larger
batches give no GPU-parallelism benefit on CPU. Smaller batches would increase noise in
gradient estimates.

**Fast dev mode (500 slices, 20 epochs)**

The full training set has ~10,000 valid slices. Full training on CPU would take 1-2 hours.
The 500-slice fast dev mode verifies the entire pipeline works, shows that loss decreases
and Dice improves, and lets you inspect predictions — all in ~10-15 minutes. Once verified,
switch to full training on GPU by setting `max_slices=None`, `batch_size=64`,
`n_epochs=100`.

**Model checkpointing**

The model with the best mean foreground Dice (average of boundary + interior Dice on
validation) is saved to disk. This is more meaningful than saving the model with lowest
loss, because loss combines CE and Dice in potentially non-intuitive ways, while foreground
Dice directly measures segmentation quality.

### Post-Processing: Semantic to Instances

The U-Net outputs a 3-class probability map. To get individual nucleus instances:

```python
def semantic_to_instances(pred_3class):
    interior = (pred_3class == 2)
    foreground = (pred_3class >= 1)  # interior + boundary
    seeds = label(interior)          # connected components → one seed per nucleus
    distance = distance_transform_edt(interior)
    instances = watershed(-distance, seeds, mask=foreground)
    return instances
```

**Step 1 — Connected components on interior class:** Each connected blob of "interior"
pixels gets a unique integer label. Because the boundary class separates touching nuclei,
each interior region corresponds to one nucleus. This is where the 3-class formulation
pays off — with binary (fg/bg), touching nuclei would be one connected component.

**Step 2 — Watershed expansion:** The connected components are used as seeds for watershed,
which expands each instance into the surrounding boundary pixels. The distance transform
of the interior provides the topography, and the mask (interior + boundary = all
foreground) constrains the expansion. This recovers the full nucleus area, including the
1-pixel boundary ring that the semantic segmentation carved out.

The key insight: the *hard* part (deciding where one nucleus ends and another begins) is
done by the neural network. The *easy* part (labeling connected regions and expanding
them) is done by classical algorithms that are fast and deterministic.

### Evaluation Metrics

**Per-class Dice score**

Overall accuracy is meaningless here (97.7% by predicting all-background). Per-class Dice
tells you how well the model segments each class independently:

- **Background Dice ~0.99:** Easy, expected to be high from the start
- **Interior Dice:** How well the model detects nuclei. The primary performance indicator.
- **Boundary Dice:** How well the model separates touching nuclei. The hardest class;
  expect 0.3-0.5, which is still useful for instance separation.

**Mean foreground Dice** (average of boundary + interior) is the single summary metric
used for model selection and checkpointing. It deliberately excludes background, which
would inflate the score.

**Instance count comparison**

After post-processing, we compare the number of predicted instances vs ground truth
instances per slice. This provides an intuitive sanity check:

- Predicted < GT → under-segmentation (missing nuclei or merging touching ones)
- Predicted > GT → over-segmentation (splitting single nuclei)
- Predicted ≈ GT → model is finding the right number of objects

This is plotted as a scatter plot (GT count vs predicted count, with a y=x reference line)
and as a histogram of count errors.

---

## 7. References

- **C. elegans dataset:** Long, F., Peng, H., Liu, X., Kim, S. K., & Myers, E. (2009).
  A 3D digital atlas of C. elegans and its application to single-cell analyses.
  *Nature Methods*, 6(9), 667-672.

- **Train/val/test split:** Weigert, M., Schmidt, U., Haase, R., Sugawara, K., & Myers, G.
  (2020). Star-convex polyhedra for 3D object detection and segmentation in microscopy.
  *IEEE/CVF WACV*, 3666-3673.

- **U-Net:** Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional
  networks for biomedical image segmentation. *MICCAI*, 234-241.

- **nnU-Net (CE + Dice loss):** Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., &
  Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based
  biomedical image segmentation. *Nature Methods*, 18(2), 203-211.

- **Watershed:** Beucher, S., & Lantuéjoul, C. (1979). Use of watersheds in contour
  detection. *International Workshop on Image Processing*.

- **Zebrahub:** Lange, M., et al. (2024). A multimodal zebrafish developmental atlas
  reveals the state-transition dynamics of late-vertebrate pluripotent axial progenitors.
  *Cell*, 187(23), 6728-6746.
