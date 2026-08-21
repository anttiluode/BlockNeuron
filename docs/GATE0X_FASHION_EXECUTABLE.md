# Gate 0X0 — executable cross-modal completion

This is the first executable bridge between the BlockNeuron line and the older Janus-style visual realization experiments.

The question is deliberately narrower than full text-to-image generation:

> Can one small shared BlockNeuron substrate learn a visual and semantic family at the same time, so that a text-only cue enters the same public state learned from images and a visual realization can be read back out?

The first dataset is Fashion-MNIST from Hugging Face (`anonyme449/fashion_mnist`): 28x28 grayscale images, ten clothing classes, 60,000 train / 10,000 test examples. The default run uses only 12,000 train and 2,000 test images so the first receipt stays small.

## Why this version is built this way

A tempting experiment is simply

```text
text embedding -> image decoder -> pixels
```

That would almost certainly work, but it would not test the BlockNeuron idea. It would be an ordinary conditional decoder with rhythmic machinery attached.

Gate 0X0 therefore has a hard anti-cheat:

```text
NO text -> pixel loss
NO text -> decoder skip
NO concept -> phase address
NO pretrained CLIP
```

Text can only reach an image through a state that has become compatible with the public state learned from the image side.

## Architecture

```text
raw text string                         28x28 image
      |                                     |
char embedding + GRU                    tiny CNN
      |                                     |
semantic receptor drive                 visual receptor drive
      \                                     /
       \                                   /
        +------ SAME RESONANT BLOCK -------+
                    |
            recurrent branch state
                    |
          +---------+---------+
          |                   |
      PUBLIC STATE        PRIVATE DETAIL
       concept-ish        instance-ish
          |                   |
          +------ coordinate decoder ------> image
          |
          +------ classifier --------------> class
```

The block has multiple learned branches. Branch gates combine:

```text
semantic receptor match
+ visual receptor match
+ modality-present receptor state
+ multi-phase affinity
```

The phase trajectory is a small trainable vector on a torus `T^K`. In this gate it is a global passing rhythm, **not** a goal-controlled SEEK/LOCK policy.

## Training operating conditions

Every batch runs the same block in three conditions.

### IMAGE ONLY

```text
image -> block -> public + detail
```

Losses:

- full reconstruction from `public + detail`;
- coarse reconstruction from `public + zero_detail`;
- image-label classification from `public`.

The coarse reconstruction is important. With private detail removed, many different images of the same class push the public-only decoder toward the stable visual family / MSE prototype.

### TEXT ONLY

```text
raw string -> block -> public + detail
```

Losses:

- align text-derived `public` with the paired image-derived `public`;
- penalize text-only private detail.

There is **no pixel target on this path** and no class-index target on this path. A text cue must learn to enter the image-trained public geometry through paired experience.

### TEXT + IMAGE

```text
text + image -> same block -> public + detail
```

Losses:

- reconstruct the actual image;
- keep the paired public state compatible with both unimodal public states;
- classify from the public state.

This is the closest toy analogue here to teaching semantic and visual receptors simultaneously on the same experience.

## Why text can make an image at inference

After training:

```text
"ankle boot"
      |
      v
text receptor -> shared block
      |
      v
public state approximately compatible with image-derived ankle-boot public states
      |
      v
private detail deliberately set to zero
      |
      v
coordinate decoder
      |
      v
class-level ankle-boot realization
```

This first gate is therefore expected to produce a **prototype/family representative**, not a unique photorealistic instance. A future instance/noise state can select different members of the family.

## Run

Install the optional cross-modal dependencies:

```bash
python -m pip install -e ".[crossmodal]"
```

Default small run:

```bash
python experiments/gate0x_fashion_crossmodal.py
```

A smaller first smoke run:

```bash
python experiments/gate0x_fashion_crossmodal.py \
  --epochs 3 \
  --train-limit 4000 \
  --test-limit 1000
```

The default is still modest:

```text
train rows      12,000
validation rows  2,000
image size          28x28
classes                 10
branches                 8
phase dimensions          4
recurrent steps           8
```

On CUDA the script uses the GPU automatically.

## What it writes

By default:

```text
runs/gate0x_fashion/
    best.pt
    last.pt
    text_prototypes.png
    text_trajectory.png
```

`text_prototypes.png` is one final text-only visual realization per class.

`text_trajectory.png` is more important: every row is a text cue and every column is one recurrent block step. It lets us see whether the image simply snaps out at the readout or whether a visual family actually develops along the internal trajectory.

## Query a trained block

```bash
python experiments/gate0x_fashion_crossmodal.py \
  --checkpoint runs/gate0x_fashion/best.pt \
  --query "ankle boot|bag|sneaker"
```

This writes:

```text
runs/gate0x_fashion/query_trajectory.png
```

The query is a raw character string. There is no lookup from the query to a phase angle.

## Metrics

The executable reports:

```text
image_to_concept_acc
text_to_concept_acc
image_recon_mse
public_only_mse
crossmodal_cosine_error
```

`text_to_concept_acc` is particularly useful because the classifier is trained on **image-derived** public state. If a text-only cue is classified correctly, it has entered a region of public state compatible with the image-trained geometry.

Pixel output is then a separate readout from that same public state.

## Mandatory controls

### 1. Clamp the rhythm and retrain

```bash
python experiments/gate0x_fashion_crossmodal.py \
  --phase-mode clamped \
  --output-dir runs/gate0x_clamped
```

If this ties the dynamic version, cross-modal completion survives but phase did not buy anything in Gate 0X0.

The script also evaluates a dynamic-trained network with phase clamped at test time. That is a mechanism ablation, not a substitute for retraining the matched clamped baseline.

### 2. Shuffle cross-modal pairing

```bash
python experiments/gate0x_fashion_crossmodal.py \
  --shuffle-pairs \
  --output-dir runs/gate0x_shuffled
```

The text marginal and image marginal remain present, but the text from one example is rolled onto another image in each batch. If correct text/image pairing is actually responsible for the shared public geometry, text-only performance should collapse or become systematically wrong.

### 3. Ordinary direct baseline

Not yet claimed by this gate. The next executable attacker should use the same tiny text encoder and coordinate decoder but replace the BlockNeuron with an ordinary MLP/GRU shared latent. If it ties or wins on all resources, Gate 0X0 remains a mechanism demonstration rather than an engineering advantage.

## Stop lines

Do not call the result interesting if any of these happens:

- text outputs look good because a hidden text-to-pixel loss slipped in;
- shuffled pairing performs as well as correct pairing;
- text-to-concept accuracy stays near chance while generated images are judged only by eye;
- the dynamic system works only because of more parameters than a matched ordinary baseline;
- phase clamp/retrain ties and we continue claiming rhythm is necessary;
- one fuzzy average prototype is presented as general text-to-image generation.

## What success would establish

A successful Gate 0X0 would establish only this:

> **The same small stateful substrate can learn visual reconstruction and cross-modal semantic alignment such that a raw text cue, without direct pixel supervision, enters an image-trained public state from which a visual family representative can be decoded.**

That is already enough to justify the next experiment.

Then SEEK/LOCK becomes meaningful:

```text
text cue
  -> block begins in a poor state
  -> controller changes phase/frequency/coupling
  -> cross-modal compatibility improves
  -> useful visual/semantic coalition locks
```

But first we make sure the animal can associate and complete at all.
