# Decision Grid Design

`DecisionScheduleConfig` is the live decision policy: it defines when a strategy
should wake up from first valid book, trade wake, top-of-book wake, or heartbeat
rules.

`decision_grid.npz` is the offline replay realization of that policy for one
execution tape. It stores the exact row sequence, tape event pointers, book
pointers, wake reasons, interval counters, the schedule payload, and a
deterministic `decision_grid_hash`.

Downstream artifacts never regenerate decisions. Ingest, linear training,
linear signals, adverse-selection training, adverse-selection signals,
execution simulation, PPO training, and policy evaluation all align rows by
`decision_grid_hash`. Artifacts without grid lineage are rejected by execution
and RL entrypoints.

Live trading will run the same scheduler online and log the same fields:
decision event key, book pointer, reason code, reason flags, elapsed time, and
event counts since the previous decision.

The ownership boundary is:

- The scheduler owns when to decide.
- Models own what to predict at a decision row.
- The policy owns what quote to send.
