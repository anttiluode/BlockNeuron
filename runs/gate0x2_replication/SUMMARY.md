# Gate 0X2 replication + ordinary attackers

The table below reports **final-epoch mean ± sample SD across seeds**.
Checkpoint selection in the underlying runs never uses held-out metrics.

| model | params | seen joint | held-out joint | held-out attr | held-out proto MSE | seen visual NN | held-out visual NN |
|---|---:|---:|---:|---:|---:|---:|---:|
| x2 | 131,320 | 1.0000 ± 0.0000 | 0.3000 ± 0.1000 | 0.8200 ± 0.1095 | 0.0267 ± 0.0048 | 0.7667 ± 0.0236 | 0.1600 ± 0.1140 |
| mlp | 130,735 | 1.0000 ± 0.0000 | 0.8200 ± 0.0837 | 1.0000 ± 0.0000 | 0.0171 ± 0.0011 | 0.7733 ± 0.0279 | 0.4400 ± 0.0548 |
| gru | 131,215 | 1.0000 ± 0.0000 | 0.6200 ± 0.0447 | 1.0000 ± 0.0000 | 0.0181 ± 0.0007 | 0.7533 ± 0.0447 | 0.4400 ± 0.0548 |

## Per-seed held-out receipt

| model | seed | selected epoch | final held-out MSE | final held-out visual NN | selected held-out visual NN |
|---|---:|---:|---:|---:|---:|
| x2 | 18001 | 16 | 0.0221 | 0.2000 | 0.2000 |
| mlp | 18001 | 13 | 0.0171 | 0.4000 | 0.4000 |
| gru | 18001 | 16 | 0.0178 | 0.4000 | 0.4000 |
| x2 | 18002 | 16 | 0.0320 | 0.1000 | 0.1000 |
| mlp | 18002 | 14 | 0.0181 | 0.4000 | 0.3000 |
| gru | 18002 | 11 | 0.0190 | 0.4000 | 0.5000 |
| x2 | 18003 | 16 | 0.0285 | 0.2000 | 0.2000 |
| mlp | 18003 | 16 | 0.0172 | 0.4000 | 0.4000 |
| gru | 18003 | 13 | 0.0171 | 0.5000 | 0.5000 |
| x2 | 18004 | 16 | 0.0297 | 0.3000 | 0.3000 |
| mlp | 18004 | 16 | 0.0177 | 0.5000 | 0.5000 |
| gru | 18004 | 15 | 0.0184 | 0.5000 | 0.5000 |
| x2 | 18005 | 16 | 0.0212 | 0.0000 | 0.0000 |
| mlp | 18005 | 16 | 0.0153 | 0.5000 | 0.5000 |
| gru | 18005 | 16 | 0.0181 | 0.4000 | 0.4000 |

## Interpretation rule

- If MLP/GRU match X2 across seeds, **factor separation** explains the X2 rescue.
- If X2 retains a reproducible held-out advantage at comparable parameter count, the structured block earns the next mechanistic attack.
- Do not choose seeds, epochs, or checkpoints by held-out performance.
