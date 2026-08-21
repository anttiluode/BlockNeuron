# Gate 0X2 — factor-separated receptor composition

Gate 0X1 answered an important question negatively. The merged text path learned all 30 seen class/attribute conjunctions, but it did not systematically produce the ten held-out visual combinations.

Full X1 receipt, seed 18001 / 12k images / 16 epochs:

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

The shuffled-attribute-word control preserved object identity but destroyed attribute meaning:

```text
heldout_class_acc         1.0000
heldout_attr_acc          0.0000
heldout_joint_acc         0.0000
heldout_visual_nn_joint   0.0000
```

So X1 learned object and quality information, but its architecture did not force those factors to remain reusable after the phrase had been collapsed into one char-GRU semantic vector.

## X2 hypothesis

Gate 0X2 makes the architectural question literal:

```text
OBJECT RECEPTOR ------------------\
                                    > SAME RECURRENT BLOCKNEURON
QUALITY RECEPTOR -----------------/              |
                                                   v
IMAGE -----------------------------> shared public visual state
                                                   |
                                                   v
                                             image decoder
```

There is no text encoder in this controlled gate. `bag` and `small` arrive through two distinct learned receptor populations. They first interact at the branch gates and branch dynamics inside the BlockNeuron.

That means the receptor for `bag` is exactly the same whether the requested combination is:

```text
small bag
large bag
left bag
right bag
```

and the receptor for `small` is exactly the same for:

```text
small bag
small sneaker
small coat
small trouser
```

The purpose is not to claim that categorical IDs are language understanding. The purpose is to isolate whether **keeping independently reusable factors separate until they meet inside the block** changes the X1 failure.

## Same held-out split

X2 uses exactly the same forbidden class/attribute pairs as X1:

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

Every class still occurs with the other three attributes. Every attribute still occurs with other classes. No held-out receptor pair occurs on any training path.

## Anti-cheat

The receptor-only path gets no direct pixel, class, or attribute target.

Training conditions are:

```text
IMAGE ONLY
image -> block -> public + private
public + private -> exact transformed instance
public           -> image-side class/attribute prototype
class/attr probes are trained here

RECEPTORS ONLY
object receptor + quality receptor -> same block -> public
public aligns to paired image-derived public state
NO pixel target
NO class target
NO attribute target

RECEPTORS + IMAGE
all three receptor families enter the same block
auxiliary paired reconstruction/alignment
```

The image decoder never receives object or attribute IDs directly.

## What would count as a rescue

The strongest X1 failure was:

```text
heldout_visual_nn_joint = 0 / 10
```

X2 evaluates the same metric. A generated receptor-only image is compared against all 40 deterministic image-side prototypes. It only scores if its nearest prototype is the exact correct unseen conjunction.

The important comparison is therefore:

```text
X1 merged phrase -> char GRU -> one semantic drive
vs
X2 object receptor + quality receptor -> block
```

If X2 preserves high seen performance and substantially improves held-out joint/visual composition, then the placement of the factor boundary matters.

If X2 also fails, then simply keeping semantic inputs separate is not enough; the image-side/public geometry or the block update itself must be made factor-compatible.

## Run

```bash
python3.13 experiments/gate0x2_receptor_composition.py
```

The default output path is control-aware:

```text
runs/gate0x2_receptor/
```

A short smoke run is available:

```bash
python3.13 experiments/gate0x2_receptor_composition.py --epochs 3 --train-limit 4000 --max-batches 30
```

## Negative control

Destroy only the relation between the quality receptor ID and the visual transformation:

```bash
python3.13 experiments/gate0x2_receptor_composition.py --shuffle-attribute-receptors
```

This automatically writes to:

```text
runs/gate0x2_receptor_attr_shuffled/
```

Unlike X1, the control-aware default prevents an accidental overwrite of the baseline directory.

## Outputs

```text
best.pt
last.pt
all_receptor_compositions.png
heldout_compare.png
heldout_trajectory.png
```

`H` marks the ten forbidden combinations in the all-compositions sheet.

## Stop line

A positive X2 result would establish only:

> Factor-separated receptor entry allows the shared BlockNeuron to reuse object and quality structure across an unseen conjunction better than the merged-text X1 path.

It would **not** establish a unique BlockNeuron advantage. The mandatory next attacker is an ordinary factorized model in which object and attribute embeddings are combined by a matched MLP/GRU/additive latent before the same decoder. If that attacker ties X2, factor separation is useful but the BlockNeuron is not uniquely responsible.
