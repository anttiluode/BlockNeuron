# Gate 0H — hysteretic branch matter

The ferroelectric inspiration is used here as a **dynamical abstraction**, not as a claim that the tensor is a crystal.

The missing ingredient is a local branch configuration that can be changed by sufficiently strong/coherent drive and then remain changed after the drive disappears.

## Continuous local state

Each branch-like material element owns a polarization-style scalar `p` in a double-well potential:

```text
U(p; E) = 1/4 (p^2 - 1)^2 - E p
```

and evolves by gradient flow:

```text
dp/dt = p - p^3 + E
```

where `E` is local effective drive.

With `E = 0`, the stable configurations are near:

```text
p = -1        p = +1
```

The software branch conductance is read from that persistent configuration:

```text
g = sigmoid(k p)
```

so the history of the drive changes the future effective branch.

## Gate 0H receipt

Two programs contain exactly the same three pulse amplitudes, only in a different order:

```text
A: +1.20, -1.20, +0.80
B: -1.20, +1.20, +0.80
```

Both start from `p = -1` and both finish with the same current field `+0.80`. After the program, the field is removed completely and the state relaxes with `E = 0`.

For the default integration constants the two histories settle in opposite wells:

```text
A -> p ≈ -1 -> low branch conductance
B -> p ≈ +1 -> high branch conductance
```

Then the external drive is absent.

The test explicitly checks:

1. **Threshold:** a weak pulse perturbs the state but it returns to the original well; a stronger pulse crosses the barrier.
2. **History dependence:** the same pulse multiset in a different order writes a different state.
3. **Retention:** after relaxation, arbitrary silent time is represented by identity on the stored configuration; no mandatory per-silent-step state transition is executed.
4. **Readout:** the persistent state changes branch conductance.
5. **Erase/rewrite:** a strong opposite pulse moves the positive state back to the negative basin.
6. **Differentiability:** before discrete sign/readout, the final continuous state has gradient with respect to the pulse amplitudes.

Run:

```bash
python3.13 experiments/gate0h_hysteretic_matter.py
```

The experiment also injects pulse noise and compares three mechanisms:

```text
Landau double-well state
explicit Schmitt latch
memoryless current-context gate
```

## Mandatory control: explicit latch

A hard Schmitt latch is deliberately included because it should solve the same write/hold/erase toy:

```text
field >= +theta -> state = +1
field <= -theta -> state = -1
otherwise       -> keep state
```

That control is important. If the continuous hysteretic state remembers traversal and a plain latch does too, then Gate 0H has **not** discovered unique memory or expressivity.

The narrower receipt is:

> A differentiable local branch configuration can be written by history-dependent drive, retained after the drive disappears, and used as a persistent conductance state.

That is the software property borrowed from ferroic matter.

## Why this is different from the current BlockNeuron gate

The existing fast gate is effectively a function of current content and current phase:

```text
g_b(t) = G(content_t, phase_t)
```

When the drive leaves, that gate leaves.

Gate 0H adds:

```text
p_b(t+1) = H(p_b(t), local_drive_t)

g_b(t) = G(content_t, phase_t, p_b(t))
```

Now a traversal can **write the future machine**.

The intended timescale ladder becomes:

```text
fast activity / phase
        ↓
fast local branch/contact state
        ↓
hysteretic configuration       <- Gate 0H
        ↓
slow learned weights
        ↓
morphology / topology
```

## What this gate does not establish

- It does not establish a BlockNeuron advantage over generic persistent state.
- It does not establish an energy advantage in software.
- It does not justify saying retention costs literally zero physical energy.
- It does not model a specific ferroelectric material quantitatively.
- It does not yet combine hysteresis with the Fashion-MNIST cross-modal system.

A clocked GRU can retain information too. An event-driven ordinary state machine can also avoid needless silent updates. Those are mandatory later resource comparisons if a stronger efficiency claim is made.

## Next combination, only after X2 replication

Keep this gate independent from the X2 factorized-composition experiment.

If the X2 result survives multi-seed replication and ordinary factorized attackers, the interesting later combination is not merely "add memory." It is:

```text
object / quality / percept drive
          ↓
shared block traverses modes
          ↓
some local p_b cross barriers
          ↓
configuration persists after cue removal
          ↓
future semantic or visual completion begins from a changed substrate
```

Then one can measure switching work, retention, interference, and whether familiar states become cheaper to re-enter without changing the slow weights.
