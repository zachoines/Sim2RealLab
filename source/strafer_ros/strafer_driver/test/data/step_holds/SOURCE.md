# Two bench holds of all four wheels

Byte-identical copies of two bench recordings, with the wheels lifted: all four motors driven
by command 37 at 535 and at 178 ticks/s for a 6 s hold, with P 15000, I 750, D 0 and QPPS 2657
on every motor. `test_tune_pid.py` fits them. The pull request that added them cites the record
they come from.

| file | bytes | sha256 |
|---|---|---|
| `all535_r1.csv` | 44051 | `3bee7014183eeec606e680d430d60e3c6ed2dc19a9eba9ec340898c75180b2c5` |
| `all178_r1.csv` | 43969 | `0d7ec690bf2dbe5e2f9a126b73289e38f6a4d0fba3ab32de59535c05376c92de` |

The columns are those the bench tool, `motor_bench.py`, writes. One row per 20 ms cycle;
`phase` is `pre` before the first command, `drive` during the hold and `tail` after it. `t` is
when the row was taken, and `tr_F` and `tr_R` are when each controller's count reads returned,
all in seconds from the first command. `c_F1` to `c_R2` are the raw counts (commands 16 and
17), `s_F1` to `s_R2` the reported speeds (commands 18 and 19), `pwm_*` the outputs (command 48)
and `v_front` and `v_rear` the battery readings. F1, F2, R1 and R2 are FL, FR, RL and RR.
