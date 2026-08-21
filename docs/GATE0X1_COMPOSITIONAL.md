# Gate 0X1 — compositional visual concepts

Gate 0X0 established the easy case: paired text and images can enter compatible public BlockNeuron state, and text alone can drive a class-level visual realization.

Gate 0X1 asks the harder question:

> Can the same tiny model reuse a learned visual quality with a learned object class in a combination it never saw during training?

## Controlled factors

Fashion-MNIST images are transformed by one of four reusable attributes:

```text
small   centered resize to 72%
large   centered resize to 122% then crop
left    shift 4 pixels left
right   shift 4 pixels right
```

Text names both factors, for example:

```text
small sneaker
right bag
large trouser
```

The character-GRU receives varied surface forms during training so the experiment is not tied to one exact sentence template.

## The crucial split

Exactly one class/attribute pair is withheld for every Fashion-MNIST class:

```text
small t-shirt top
large trouser
left pullover
right dress
small coat
large sandal
left shirt
right sneaker
small bag
large ankle boot
```

Those ten combinations never occur in training.

Every class is still present with the other three attributes, and every attribute is present with several other classes. Therefore the model cannot solve the held-out test by merely memorizing the 30 observed class/attribute phrases.

## Architecture

The architecture is intentionally the same small Gate 0X0 system:

```text
raw text -> tiny char GRU ----\
                              -> SAME BlockNeuron -> public + private state
image -> tiny CNN -----------/                         |
                                                coordinate decoder
```

A tiny image-trained attribute head is added only as a probe of the public state.

## Public versus private visual state

For X0, the public-only decoder was trained against the current image and naturally learned blurry class averages.

For X1 the target is made explicit. The training subset first produces ten image-side class means. Each mean is transformed by all four attributes, producing a 10 x 4 bank of deterministic visual-family prototypes.

Training then asks:

```text
image public + private -> exact transformed instance
image public           -> transformed class/attribute prototype
```

Text-only state receives no direct pixel, class, or attribute target. It is aligned to the public state reached by its paired image.

The paired text+image path remains an auxiliary training condition, but **no held-out class/attribute combination appears on any training path**.

## What counts as composition

At evaluation the script queries all 40 canonical prompts. Thirty are seen combinations and ten are held out.

The held-out metrics are the gate:

```text
heldout_class_acc
heldout_attr_acc
heldout_joint_acc
heldout_proto_mse
heldout_visual_nn_joint
```

`heldout_joint_acc` asks whether the image-trained class and attribute readouts both identify the text-only public state correctly.

`heldout_visual_nn_joint` is harder and does not trust those heads: the generated image is compared against all 40 image-side prototypes, and the nearest prototype must be the exact unseen class/attribute combination.

## Full-run receipt

The intended 12,000-image / 16-epoch run completed on seed 18001:

```text
seen_joint_acc            1.0000
heldout_joint_acc         0.2000
heldout_class_acc         0.7000
heldout_attr_acc          0.4000
seen_proto_mse            0.0079
heldout_proto_mse         0.0571
seen_visual_nn_joint      0.7667
heldout_visual_nn_joint   0.0000
```

The important pattern is not undertraining. Seen joint state reaches 100% and seen visual nearest-neighbour composition reaches 23/30, while the ten held-out visual combinations remain 0/10.

The shuffled-attribute-word control preserves class semantics but destroys quality semantics:

```text
seen_joint_acc            0.0000
heldout_joint_acc         0.0000
heldout_class_acc         1.0000
heldout_attr_acc          0.0000
seen_visual_nn_joint      0.0667
heldout_visual_nn_joint   0.0000
```

### Verdict

**Factor acquisition: PASS. Seen conjunctions: PASS. Systematic visual recombination: FAIL.**

The merged text path learned object identity and attribute meaning, but it did not spontaneously make those factors reusable in unseen visual conjunctions. More training is not the obvious remedy: seen composition improved strongly while held-out visual composition stayed at zero.

This failure motivates Gate 0X2. In X1, `small bag` is collapsed by the char-GRU into one semantic vector before the BlockNeuron ever sees it. X2 keeps object identity and quality as separate receptor populations until they meet inside the shared block.

See [`GATE0X2_RECEPTOR_COMPOSITION.md`](GATE0X2_RECEPTOR_COMPOSITION.md) on the X2 branch / PR #3.

## Visual receipts

The default run writes:

```text
runs/gate0x1_compositional/
    best.pt
    last.pt
    all_compositions.png
    heldout_compare.png
    heldout_trajectory.png
```

`all_compositions.png` is a 10 x 4 grid. Cells marked `H` are combinations never shown during training.

`heldout_compare.png` places each generated held-out image beside its deterministic image-side target.

`heldout_trajectory.png` shows the eight recurrent BlockNeuron steps for the ten unseen text prompts.

The committed full receipts are under `runs/gate0x1_full/` and `runs/gate0x1_full_attr_shuffled/`.

## Run

From the repository root on the compositional branch:

```bash
python3.13 experiments/gate0x1_fashion_compositional.py
```

The default remains small:

```text
train images     12,000
base model       140,236 parameters
extra attr head      100 parameters
attributes             4
seen combinations     30
held-out combinations 10
recurrent steps         8
epochs                  16
```

A quick smoke run:

```bash
python3.13 experiments/gate0x1_fashion_compositional.py --epochs 3 --train-limit 4000 --max-batches 30
```

## Negative control

Destroy only the meaning of the attribute word while preserving the images, class words and visual transformations:

```bash
python3.13 experiments/gate0x1_fashion_compositional.py \
  --shuffle-attribute-words \
  --output-dir runs/gate0x1_attr_shuffled
```

If held-out attribute/composition performance remains strong under that control, the interpretation is wrong.

## Stop line

Gate 0X1 did **not** satisfy its strong success criterion. It did establish that the system can learn the individual factors and all seen cross-modal conjunctions, while exposing the merged-text bottleneck as a plausible reason systematic visual recombination failed.

No BlockNeuron advantage follows from X1. A matched ordinary factorized/recurrent attacker remains required for any later positive composition result.
