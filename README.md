# BlockNeuron

`BlockNeuron` is an exploratory research repo asking a simple but slippery question:

> What becomes possible if the basic learned unit is not a memoryless point, but a small structured substrate with compartments, local state, modulation, rhythm, and eventually persistent reconfiguration?

This repo is **not** a claim that the code is a faithful neuron model, a new theory of the brain, or a better replacement for ordinary neural networks. Most of the work is organized as small gates on feature branches. A gate is useful even when it kills the idea that motivated it.

The main branch is intentionally a map. The experimental code and receipts live on the branches below.

## What happened on 2026-08-21

A lot was built in one day, but the result is narrower than the amount of code might suggest.

We established several mechanisms that the BlockNeuron-style substrate can implement. We **did not** establish a general performance advantage over ordinary neural networks. In the most direct matched test so far, ordinary factorized MLP/GRU models beat the BlockNeuron on held-out compositional generalization.

The clearest lessons so far are:

- **Structured gating works as an instrument.** The same fixed parameters can expose different effective computations under modulation/phase, and ordered phase traversal can matter. Ordinary conditioned/recurrent attackers can do this too.
- **Cross-modal completion works in a tiny toy system.** A ~140k-parameter shared text/image model can learn Fashion-MNIST concept families and reconstruct a coarse visual family from a text cue without direct text-to-pixel supervision. However, a separately trained phase-clamped model performs about as well: rhythm is being used, but is not required for this task.
- **Systematic composition did not become a BlockNeuron advantage.** Merging `small bag` into one text vector produced poor held-out visual composition. Keeping `small` and `bag` as separate receptors helped, but five-seed matched controls showed that ordinary factorized MLP/GRU fusion works substantially better. The useful result is **factor separation**, not BlockNeuron routing.
- **Hysteretic local state works.** A continuous double-well state can be written, retained after the drive disappears, alter branch conductance, and be erased/re-written. This is a useful differentiable software primitive, but an explicit latch ties the memory capability.
- **A configured computation can persist.** A brief write can select COPY / NEGATE / ROTATE / FILTER, after which the programming signal is removed and the operation can be reused with zero mandatory material-state updates during silence. Again, an event-driven latch/register can do the same job more simply.
- **Tiny TSP works as a side experiment.** A city-by-tour-position hysteretic field can settle to exact tours on the tiny deterministic CI cases. The plain continuous relaxer and nearest-neighbour + 2-opt also solve them exactly, and the current readout uses an exact tiny-instance permutation projection. This is a capability demonstration, not a TSP result.

So the repo has **not** discovered a superior new neuron. It has produced a set of executable probes for a different computational organization: fast activity, slower local configuration, and still slower learned structure. The most interesting surviving question is whether that middle timescale can do something useful that a compact explicit state register does not already do better.

## Branch map

```text
main
│
└── sol/gate0-same-mass-different-machine        PR #1
    │
    │   Gate 0   same mass / different effective machine
    │   Gate 0R  ordered passing modes
    │   Gate 0X0 tiny text↔image shared-state completion
    │   Gate 0S  SEEK/LOCK design notes
    │
    └── sol/gate0x1-compositional                PR #2
        │   controlled Fashion-MNIST qualities
        │   held-out adjective+noun combinations
        │   visual systematic composition: FAIL
        │
        └── sol/gate0x2-receptor-composition     PR #3
            │   keep object and quality separate until the block
            │   first seed: partial rescue
            │
            ├── sol/gate0x2-replication-attacker PR #4
            │   five seeds + matched factorized MLP/GRU
            │   verdict: ordinary factorization wins
            │
            └── sol/gate0h-hysteretic-matter     PR #5
                │   continuous double-well branch configuration
                │   write / hold / erase
                │
                └── sol/gate0h1-configured-substrate  PR #6
                    │   configure computation once, reuse later
                    │   latch is mandatory control and ties capability
                    │
                    └── sol/gate0t-tsp            PR #7
                        tiny TSP side quest on hysteretic matter
```

### Branch links

| Branch | Question | Current verdict |
|---|---|---|
| [`sol/gate0-same-mass-different-machine`](https://github.com/anttiluode/BlockNeuron/tree/sol/gate0-same-mass-different-machine) | Can fixed structure expose different effective computations under modulation/rhythm, and can text/image experience meet in one small shared substrate? | Mechanisms **yes**; unique advantage **no**. |
| [`sol/gate0x1-compositional`](https://github.com/anttiluode/BlockNeuron/tree/sol/gate0x1-compositional) | Does the shared substrate recombine visual concepts it never saw together? | Factor meanings learned; systematic held-out visual composition **failed**. |
| [`sol/gate0x2-receptor-composition`](https://github.com/anttiluode/BlockNeuron/tree/sol/gate0x2-receptor-composition) | Does keeping object and quality as separate receptors help? | **Partial rescue** on the first seed. |
| [`sol/gate0x2-replication-attacker`](https://github.com/anttiluode/BlockNeuron/tree/sol/gate0x2-replication-attacker) | Is that rescue actually a BlockNeuron effect? | **No.** Matched factorized MLP/GRU beat X2 across five seeds. |
| [`sol/gate0h-hysteretic-matter`](https://github.com/anttiluode/BlockNeuron/tree/sol/gate0h-hysteretic-matter) | Can local configuration cross a barrier and persist after the drive disappears? | **Yes as a mechanism.** Explicit latch ties memory capability. |
| [`sol/gate0h1-configured-substrate`](https://github.com/anttiluode/BlockNeuron/tree/sol/gate0h1-configured-substrate) | Can a brief write leave a reusable computation behind? | **Yes as an organizational mechanism.** No advantage over an event-driven latch/register established. |
| [`sol/gate0t-tsp`](https://github.com/anttiluode/BlockNeuron/tree/sol/gate0t-tsp) | Can the hysteretic substrate host a tiny combinatorial optimization problem? | **Yes, tiny capability only.** Ordinary controls tie; current projection is a crutch. |

## A few concrete receipts

### X1 → X2 → ordinary attackers

The composition line ended with the most useful negative result of the day. Five-seed means on the ten held-out class/quality combinations:

```text
model    heldout_joint   heldout_attr   heldout_proto_MSE   heldout_visual_NN
X2       .300 ± .100     .820 ± .110    .0267 ± .0048       .160 ± .114
MLP      .820 ± .084    1.000 ± .000    .0171 ± .0011       .440 ± .055
GRU      .620 ± .045    1.000 ± .000    .0181 ± .0007       .440 ± .055
```

All three master the seen combinations. The difference is held-out recombination. The branchy/rhythmic machinery is not helping here; it appears to encourage conjunction-specific specialization where a simpler factorized representation stays more systematic.

### Hysteretic branch state

The software abstraction uses a double-well potential:

```text
U(p; E) = 1/4 (p^2 - 1)^2 - E p
dp/dt   = p - p^3 + E
```

A sufficiently strong/history-dependent field can move `p` between wells. Once settled, the abstraction permits silent time to pass with no mandatory state transition. This is **not** a claim of zero physical energy or a faithful ferroelectric material model.

### H1 persistent computation

A four-element local configuration selects one of four fixed operations. CI showed exact hard routing, soft-route MSE around `1e-6`, retention across 10,000 silent ticks with zero executed material-state updates, and successful reprogramming followed by a 1,000,000-tick silent interval.

The numerical Landau write/relaxation costs hundreds of scalar integration updates in software. The useful property is not a cheap simulated write; it is that the write can be amortized over many later uses. An explicit latch has the same write-once organization and remains the strongest simple attacker.

### Tiny TSP

The TSP branch uses an `n × n` city-by-position field, a Sinkhorn soft permutation, and tour-cost gradients as a field on the double-well state. On the deterministic 6-city CI receipt, hysteretic relaxation, plain relaxation, and NN+2-opt all hit the exact optimum on 3/3 instances. The current final legal-tour projection is factorial and deliberately limits the claim to tiny instances.

## Research ideas that survived the day

The project started with a biological-looking picture — dendrites, oscillations, modes, persistent compartments — but the experiments are pushing the useful abstraction toward **structured dynamic matter** rather than “a more realistic neuron.”

The working timescale picture is:

```text
fast                 medium                         slow
activity / signal -> local configuration / state -> weights / structure
```

That middle layer is the open question.

### H2 — graded history-shaped reconfiguration

H0/H1 are still fancy latches unless continuous local state buys something. The next serious gate should test whether repeated experience reshapes a **graded local configuration landscape** such that related future computations require less reconfiguration than unrelated ones.

A possible quantity is:

```text
D(A, B) = reconfiguration work required to move substrate A -> B
```

The important control is an explicit compact register/latch with whatever geometry is needed to be fair. If the same behavior is trivial once a register is given a hand-designed metric, the hysteretic story has not earned anything.

### Persistent concept / percept configuration

Later, and only after H2 is clean, a semantic or sensory episode could write a temporary persistent branch configuration:

```text
cue / experience
      ↓
local configuration changes
      ↓
cue disappears
      ↓
related future processing starts from a different substrate
```

This is closer to “experience changes which computation is cheap next” than to ordinary hidden-state recurrence. It should be attacked by explicit caches, latches, fast weights, recurrent state, and event-driven alternatives.

### SEEK / LOCK remains unfinished

The older SEEK/LOCK idea is still conceptually interesting: a system should autonomously search phase/frequency/coupling from match/error rather than being handed the correct phase address. Nothing today established that capability. It remains a design direction, not a result.

### TSP without the projection crutch

If the TSP side quest continues, the next version should make city uniqueness / position uniqueness / subtour pressure part of the local dynamics and remove factorial final projection. The interesting question would then be whether a legal tour can **crystallize from local interactions** rather than being repaired by an exact readout.

## Scientific stop lines

Do not infer from this repo that:

- biological neurons literally implement these equations;
- oscillations are necessary for the demonstrated cross-modal behavior;
- hysteresis creates memory that ordinary state machines cannot represent;
- the current software model saves physical energy;
- BlockNeuron improves systematic compositionality — the matched ordinary attackers currently say the opposite;
- tiny exact TSP results imply useful scaling or a new optimization method;
- a successful toy gate is a novelty claim.

The repo is most useful when a branch can end with **“the ordinary attacker wins.”** That tells us which part of the story was decorative and which part was actually doing work.

## Current direction

The strongest surviving program is therefore not “make a neuron with more biological features.” It is:

> Build a small computational substrate with multiple timescales, then ask whether persistent local configuration changes the cost, locality, interference, or reuse of future computation in a way that survives simple explicit-state controls.

That is narrower than where the day started, but it is also much easier to falsify.
