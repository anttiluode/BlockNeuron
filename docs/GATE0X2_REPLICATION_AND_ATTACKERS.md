# Gate 0X2 — replication and mandatory ordinary attackers

The first full Gate 0X2 run is encouraging but is **one seed and ten held-out conjunctions**, so it is not yet a stable architectural result.

Seed `18001`, 12,000 Fashion-MNIST training images, 16 epochs:

```text
X1 merged-text baseline
heldout_joint_acc         0.2000
heldout_attr_acc          0.4000
heldout_proto_mse         0.0571
heldout_visual_nn_joint   0.0000

X2 factor-separated receptors
seen_joint_acc            1.0000
heldout_joint_acc         0.3000
heldout_class_acc         0.5000
heldout_attr_acc          0.7000
seen_proto_mse            0.0075
heldout_proto_mse         0.0215
seen_visual_nn_joint      0.7667
heldout_visual_nn_joint   0.2000
```

The matched shuffled-quality-receptor control ended at:

```text
heldout_class_acc         1.0000
heldout_attr_acc          0.0000
heldout_joint_acc         0.0000
heldout_proto_mse         0.0923
heldout_visual_nn_joint   0.0000
```

The single-seed signal is therefore:

- seen performance is retained;
- held-out prototype error drops strongly relative to X1;
- exact held-out visual nearest-neighbour composition moves from `0/10` to `2/10`;
- quality semantics disappear when quality receptor identity is deliberately mismatched.

This is enough to justify replication, not enough to claim a BlockNeuron advantage.

## Mandatory question

X1 and X2 differ in a simple way that ordinary networks can also exploit:

```text
X1
"small bag" -> one char-GRU vector -> shared block

X2
BAG factor -----\
                 > shared block
SMALL factor ---/
```

So the next question is:

> Is the X2 rescue caused by the BlockNeuron, or simply by keeping object and quality factorized?

Two ordinary attackers are provided.

## Attacker A — factorized MLP

```text
object embedding ----\
attribute embedding --+--> concatenate --> MLP --> public/private state
image encoding -------/
```

No branches. No receptor bank. No phase. No recurrent BlockNeuron state.

The default hidden width is chosen so the total model parameter count is close to X2's `131,320` parameters.

Run one seed:

```bash
python3.13 experiments/gate0x2_factorized_attacker.py --attacker mlp
```

## Attacker B — factorized GRU

```text
object token
attribute token
image token
     |
ordinary GRU
     |
public/private state
```

Again there are no BlockNeuron branches or phase variables. The GRU hidden size is chosen to stay near the same total parameter budget.

Run one seed:

```bash
python3.13 experiments/gate0x2_factorized_attacker.py --attacker gru
```

Both attackers use the **same**:

- Fashion-MNIST subset;
- ten held-out class/attribute pairs;
- transformed image targets;
- image encoder;
- public/private dimensionality;
- coordinate image decoder;
- image-only supervision;
- semantic-to-image-public alignment loss;
- class/attribute probes;
- optimizer and default epoch count;
- held-out metrics.

The semantic-only attacker path, like X2, receives no direct pixel/class/attribute target.

## Five-seed replication suite

The default replication suite runs:

```text
seeds: 18001, 18002, 18003, 18004, 18005
models: X2, factorized MLP, factorized GRU
```

That is 15 small training runs.

```bash
python3.13 experiments/gate0x2_replication_suite.py
```

Outputs:

```text
runs/gate0x2_replication/
    x2/seed_18001/...
    x2/seed_18002/...
    ...
    mlp/seed_18001/...
    ...
    gru/seed_18001/...
    ...
    summary.csv
    summary.json
    SUMMARY.md
```

For a quick machinery test before the full suite:

```bash
python3.13 experiments/gate0x2_replication_suite.py \
  --seeds 18001 \
  --models x2,mlp,gru \
  --epochs 2 \
  --train-limit 2000 \
  --max-batches 10
```

## Selection discipline

The underlying training scripts save `best.pt` only by a **seen-only** score:

```text
seen_joint_acc
+ seen_visual_nn_joint
- seen_proto_mse
```

Held-out performance is never used for model selection.

The replication summary reports both final-epoch and seen-selected checkpoint metrics. The main cross-seed table uses final epochs to keep the comparison simple and fixed.

## Decision rule

### Ordinary attackers tie X2

If MLP/GRU match X2 on held-out prototype MSE and held-out visual composition across seeds:

> **Factor separation explains the rescue. BlockNeuron is not needed for Gate 0X2 composition.**

That is still a useful result: X1 failed because it collapsed the semantic factors too early.

### X2 remains reproducibly better

If X2 retains a meaningful held-out advantage across seeds at comparable parameter count:

> The structured recurrent branch system has earned a narrower mechanistic follow-up.

That would still not prove biological relevance or novelty. It would justify attacks on which X2 ingredient matters: branches, recurrence, phase, receptor scoring, or their interaction.

## What comes after this

Do **not** add hysteresis before this comparison is settled.

The ferroelectric-inspired / hysteretic state is a separate hypothesis: a local branch configuration can cross a threshold, persist without mandatory per-step refresh, and alter what future modes are easy to enter. That deserves its own Gate 0H after the X2 factorization question has a multi-seed answer.
