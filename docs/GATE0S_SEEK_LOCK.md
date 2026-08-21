# Gate 0S — SEEK / LOCK

## Hypothesis

Phase should not be handed to the system as an address.

A stronger mechanism is goal-conditioned control of an internal oscillator:

```text
what I want
    +
what I am currently getting
    -> change phase / frequency / coupling
    -> pass through another effective regime
    -> detect better match
    -> slow / lock / exploit
```

The oscillator is then part of the policy, not merely a clock.

## Minimal equations

Let the block expose a smooth family of effective operators

```text
W_eff(phi, m)
```

and let a stateful controller modify its own rhythmic state:

```text
dphi/dt   = omega

domega/dt = pi(q, h, local_match, omega)
```

where

```text
q            current goal / query
h            local persistent state
local_match  evidence that the current regime is useful
pi           a small learned controller
```

The controller is not given the correct phase label.

A useful qualitative policy would have three regimes:

```text
SEEK     move through modes while match is poor
LOCK     reduce phase velocity / synchronize when match rises
RELEASE  resume movement when the regime stops being useful
```

The important claim is not that the system is conscious or that biological oscillations literally implement this algorithm. The research question is whether an artificial structured unit can learn such a closed-loop rhythmic control law and whether it buys anything measurable.

## Why this is biologically plausible but not established

Neuroscience already supports several weaker pieces:

- attention and cognitive control are associated with task-dependent changes in oscillatory synchrony, phase resetting, entrainment and cross-frequency coupling;
- prefrontal activity can bias long-range oscillatory coordination toward task-relevant sensory representations;
- hippocampal encoding and retrieval are associated with different theta phases;
- neurofeedback experiments show that humans can, under some conditions, learn to modulate oscillatory power/frequency/synchrony, although recent controlled work also warns that apparent voluntary control can be confounded by nonspecific time-on-task effects.

Therefore `goal -> altered rhythmic state -> altered effective computation` has biological precedent. `the brain deliberately searches phase space for an answer` remains a hypothesis, not an established fact.

## First artificial test

Use a fixed smooth phase-dependent latent manifold or microcircuit with overlapping modes.

A target is specified by **content**, not by a phase index. Start the oscillator at a random phase.

The controller receives only:

```text
query / target content
current block output or local match signal
current phase velocity / local state
```

It must learn to steer its own phase dynamics until the current functional regime produces a good match.

### No-cheat variants

The strongest version should progressively remove direct addressing information:

1. `CONTENT`: controller sees target content and current output.
2. `MATCH`: controller sees only a scalar/vector match signal plus its recent change.
3. `LOCAL`: each branch sees only its local match/receptor state; no global target phase or dense error vector.

The interesting version is `LOCAL`.

## Mandatory attackers

```text
DIRECT-ADDRESS      MLP predicts the best phase directly from target content
FOURIER-RNN         generic time/phase-conditioned recurrent controller
GRU / SSM           matched-state ordinary recurrent policy
FIXED-SWEEP         oscillator scans phase at constant frequency
RANDOM-SWEEP        random frequency / phase resets
ORACLE-GRADIENT     direct gradient ascent on target match, when differentiable
```

If DIRECT-ADDRESS is cheaper and equally robust, SEEK/LOCK is only an interesting mechanism, not an engineering win.

## Metrics

Do not score only final accuracy.

```text
success / retrieval error
steps until useful regime is found
control energy / phase adjustment
number of regimes visited
persistent state bytes
executed work / memory traffic
robustness to phase noise and timing jitter
generalization to unseen target contents
recovery when target changes mid-cycle
```

A particularly interesting quantity is **search amortization**:

```text
first encounter cost
vs
cost after slow structure adapts to repeated target families
```

If repeated experience makes a useful regime easier to find or lock onto, Gate 0S begins to connect directly to Gate 5.

## Relation to Janus

Janus should not be interpreted as a filing cabinet of phase slots.

Its smeared intermediate images suggest a smooth trajectory through overlapping realizations. In SEEK/LOCK language:

```text
phase sweep = moving through candidate realizations
bleed       = neighboring regimes coexist
match       = current realization resembles the sought content
lock        = stop treating the phase axis as passive; actively dwell where useful
```

This suggests a future Janus experiment in which a target image or feature vector is supplied and a learned phase controller must navigate the existing complex network to a phase region that best reconstructs the target, without being told the target angle.

## Stop line

Kill the stronger interpretation if any of the following is true:

- the learned controller merely reconstructs a hidden phase label from the query;
- fixed sweeping performs just as well at the same latency/energy;
- a direct phase-address MLP is strictly cheaper and equally robust;
- apparent SEEK/LOCK disappears when phase labels and global error vectors are removed;
- slow structural adaptation does not reduce future search cost.

## Working statement

> **A useful oscillator would not simply replay a rhythm. It would learn how to alter its own phase/frequency/coupling so that a desired temporary computational regime becomes reachable, recognizable, and stable long enough to use.**
