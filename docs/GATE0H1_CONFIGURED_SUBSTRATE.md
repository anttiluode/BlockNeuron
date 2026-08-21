# Gate 0H1 — computation leaves a configuration behind

Gate 0H established the primitive: a continuous double-well branch state can be written by traversal history, retained after the drive disappears, and erased/re-written. An explicit latch tied the memory capability, so the useful next question is not “can hysteresis remember?”

Gate 0H1 asks:

> Can a brief programming event configure which computation is available, remove the program completely, and then reuse the configured machine many times without maintaining the configuration through recurrent updates?

This is a test of **temporal organization and amortization**, not unique expressivity and not physical energy.

## Substrate

There are four fixed 4D operations:

```text
COPY
NEGATE
ROTATE
FILTER
```

The hysteretic substrate has one continuous branch-configuration scalar per operation. Programming operation `k` applies one local field vector:

```text
target branch : +1.2
other branches: -1.2
```

The branch states follow the same Gate 0H double-well dynamics:

```text
U(p; E) = 1/4 (p^2 - 1)^2 - E p
dp/dt   = p - p^3 + E
```

After relaxation one branch sits near `+1` and the others near `-1`. Conductance reads that persistent configuration and routes later data through the selected operation.

Critically, after programming:

```text
program = absent
material-state update = absent
input x -> already-configured route -> output y
```

The same state can be held across an arbitrary silent horizon in the ideal nonvolatile abstraction without executing another state transition.

## Controls

### Explicit latch

A one-hot operation latch is written once and retained with zero updates. It should tie H1 capability exactly.

If it does, H1 must **not** claim unique memory or expressivity.

### Context mux

A memoryless mux computes the exact same operation, but the operation context must be supplied on every read.

This gives a simple control-bandwidth comparison:

```text
write-once substrate: 4 control scalars once
explicit latch:       4 control scalars once
context mux:          4 control scalars every read
```

### Clocked state keeper

An exact one-hot state is retained, but a conventional clocked implementation performs an identity state transition on every read. This is only an accounting contrast. An event-driven keeper can skip that update and then reduces to the latch control.

## What is measured

The executable receipt checks:

- all four operations can be written and read;
- the program is absent during later data reads;
- the selected operation survives 10,000 silent ticks with zero material-state updates;
- direct reprogramming works;
- the soft differentiable conductance read remains close to the hard selected operation;
- noisy programming still lands in the requested configuration;
- one-time control/write costs amortize over repeated use.

The experiment reports two separate resource proxies:

1. **external control values** — how much configuration information must be supplied after the initial write;
2. **software integration scalar updates** — the numerical work used to simulate the Landau write/relaxation.

Neither is a physical joule measurement.

## Run

```bash
python3.13 experiments/gate0h1_configured_substrate.py
```

Smaller CI-style receipt:

```bash
python experiments/gate0h1_configured_substrate.py --samples 256 --trials 100
```

## Expected accounting shape

For `N` uses of one configuration:

```text
H external control / use      = 4 / N
latch control / use           = 4 / N
context control / use         = 4
clocked state updates / use   = 1
```

The simulated hysteretic write has a larger one-time numerical integration cost than the latch. That cost is explicitly reported and likewise amortizes as `1/N`.

The meaningful H1 claim, if the receipt passes, is therefore narrow:

> A computation can be written into persistent local configuration once and then reused without replaying its program or executing mandatory maintenance transitions during silence.

The explicit latch demonstrates that this organizational property is not unique to hysteresis.

## Stop line

Do not claim:

- physical energy savings;
- zero write cost;
- unique memory;
- unique computational expressivity;
- an advantage over event-driven digital latches.

A stronger future gate would need a reason why the **continuous, locally writable, branch-coupled configuration** is useful beyond what a compact explicit state register provides — for example, learned local reconfiguration, graceful partial switching, interference structure, or integration with branch growth/plasticity.
