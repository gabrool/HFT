# Decision Grid Design

`DecisionScheduleConfig` is the decision policy: it defines when a strategy
should wake up from first valid book, trade wake, top-of-book wake, or heartbeat
rules. The default schedule is an unthrottled event-driven replay grid:
`min_decision_interval_us=0`, `max_decision_interval_us=500_000`,
`wake_on_trade=true`, `wake_on_top_of_book=true`, and
`l1_size_change_fraction=0.0`, where any actual L1 size change arms a
top-of-book wake.

`decision_grid` is the offline replay realization of that policy for one
execution tape. It stores the exact row sequence, tape event pointers, book
pointers, wake reasons, interval counters, the schedule payload, and a
deterministic `decision_grid_hash`.

Downstream artifacts never regenerate decisions. Ingest, linear training,
linear signals, adverse-selection training, adverse-selection signals,
execution simulation, PPO training, and policy evaluation all align rows by
`decision_grid_hash`. Execution and RL entrypoints require the current grid
lineage fields.

Throttled schedules remain available by explicitly passing
`--min-decision-interval-us`, and fixed-grid ablations can still exist as
schedule configurations. They must be realized into a `decision_grid` before
downstream stages consume them.

Live trading will run the same scheduler online and log the same fields:
decision event key, book pointer, reason code, reason flags, elapsed time, and
event counts since the previous decision.

Live trading still has compute latency, order latency, exchange limits, and
policy-level rate constraints. Those limits should be modeled in execution
simulation or policy constraints, not by hiding market events from the default
decision grid.

The ownership boundary is:

- The scheduler owns when to decide.
- Models own what to predict at a decision row.
- The policy owns what quote to send.
