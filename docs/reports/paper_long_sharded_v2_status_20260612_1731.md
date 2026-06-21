# Paper-long Sharded V2 运行状态报告

生成时间：2026-06-12 17:31

输出根目录：

`outputs/paper_training_long_dispatch_set_attention_v1_happo_20260612_sharded_v2`

---

## 1. 当前运行方式

当前运行的是独立实验 shard，并不是共享权重并行 rollout。

已启动 12 个独立 HAPPO shard：

- seed 9401: `base`, `lower_lr`, `higher_entropy`, `larger_network`
- seed 9402: `base`, `lower_lr`, `higher_entropy`, `larger_network`
- seed 9403: `base`, `lower_lr`, `higher_entropy`, `larger_network`

每个 shard 都会训练出独立 checkpoint。该组实验适合短期调参侦察和多 seed 对照，不适合作为“集中训练一套主模型”的最终加速方案。

---

## 2. 健康状态

12 个 tmux session 全部存活：

- `pl2_sa_9401_base`
- `pl2_sa_9401_lower_lr`
- `pl2_sa_9401_higher_entropy`
- `pl2_sa_9401_larger_network`
- `pl2_sa_9402_base`
- `pl2_sa_9402_lower_lr`
- `pl2_sa_9402_higher_entropy`
- `pl2_sa_9402_larger_network`
- `pl2_sa_9403_base`
- `pl2_sa_9403_lower_lr`
- `pl2_sa_9403_higher_entropy`
- `pl2_sa_9403_larger_network`

错误日志扫描结果：

- 未发现 `Traceback`
- 未发现 `RuntimeError`
- 未发现 `CUDA out of memory`
- 未发现普通 `Error`

---

## 3. 首轮 Episode / Update 结果

所有 12 个 shard 均已完成第 1 个 episode，并进入第 2 个 episode。

首轮 HAPPO 日志如下：

| Shard | Episode 1 reward | Cost | Violations | Projection gap MW | Critic loss | DSO loss | Dispatch loss | Portfolio loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 9401 base | -447.6965 | 27659.8948 | 0 | 0.000000 | 0.010006 | -1.380362 | -0.095806 | -0.010761 |
| 9401 lower_lr | -447.6965 | 27659.8948 | 0 | 0.000000 | 0.010006 | -1.380362 | -0.293399 | -0.011426 |
| 9401 higher_entropy | -447.6965 | 27659.8948 | 0 | 0.000000 | 0.010006 | -4.141086 | -0.166512 | -0.032700 |
| 9401 larger_network | -501.8949 | 29339.2583 | 0 | 0.000000 | 0.009269 | -1.380362 | -0.030678 | -0.009783 |
| 9402 base | -716.4415 | 32992.0971 | 0 | 0.000000 | 0.012314 | -1.380362 | -0.034502 | -0.011121 |
| 9402 lower_lr | -716.4415 | 32992.0971 | 0 | 0.000000 | 0.012314 | -1.380362 | -0.020232 | -0.011023 |
| 9402 higher_entropy | -716.4415 | 32992.0971 | 0 | 0.000000 | 0.012314 | -4.141086 | -0.105237 | -0.033051 |
| 9402 larger_network | -769.3853 | 34830.5123 | 0 | 0.000000 | 0.010909 | -1.380362 | -0.035368 | -0.009552 |
| 9403 base | -604.6740 | 30335.7458 | 0 | 0.000000 | 0.010492 | -1.380362 | -0.042814 | -0.011129 |
| 9403 lower_lr | -604.6740 | 30335.7458 | 0 | 0.000000 | 0.010492 | -1.380362 | -0.577246 | -0.011258 |
| 9403 higher_entropy | -604.6740 | 30335.7458 | 0 | 0.000000 | 0.010492 | -4.141086 | -0.113549 | -0.033067 |
| 9403 larger_network | -654.9300 | 31910.9476 | 0 | 0.000000 | 0.009312 | -1.380362 | -0.035368 | -0.009501 |

---

## 4. 当前第 2 个 Episode 进度

17:31 左右的进度快照：

| Shard | 当前 episode | 当前 step | 当前 reward |
|---|---:|---:|---:|
| 9401 base | 2 | 96 | -46.7723 |
| 9401 lower_lr | 2 | 120 | -85.9312 |
| 9401 higher_entropy | 2 | 120 | -79.6599 |
| 9401 larger_network | 2 | 120 | -103.4315 |
| 9402 base | 2 | 120 | -130.6543 |
| 9402 lower_lr | 2 | 96 | -104.3076 |
| 9402 higher_entropy | 2 | 144 | -157.3060 |
| 9402 larger_network | 2 | 96 | -112.9335 |
| 9403 base | 2 | 72 | -83.2753 |
| 9403 lower_lr | 2 | 72 | -84.7378 |
| 9403 higher_entropy | 2 | 72 | -83.2537 |
| 9403 larger_network | 2 | 72 | -91.7701 |

---

## 5. 资源使用情况

GPU：NVIDIA A800 80GB

17:31 左右显存：

- 总显存：81920 MiB
- 已用显存：75730 MiB
- 其中外部 ASR 服务约 4898 MiB
- 每个普通 HAPPO shard 约 5836 MiB
- 每个 larger_network shard 约 6072 MiB

结论：

当前 12 shard 已接近显存安全上限。不能继续扩到 16 或 20 shard，否则多个 shard 同时 update 时很容易触发 CUDA OOM。

CPU：

- 每个 shard 约 150% CPU
- 每个 shard 约 12 个线程
- 12 shard 合计约 18 个 CPU core 左右
- CPU 仍有余量，但 GPU update 峰值已成为更重要的约束

---

## 6. 初步调参判断

仅基于第 1 个 episode，不能判断收敛优劣，只能作为早期风险信号。

当前可见现象：

1. `violations=0` 且 `projection_gap_mw=0`，说明安全投影/AC-aware DOE 目前仍在兜底，安全指标没有爆。
2. `larger_network` 在 3 个 seed 上的 episode 1 reward 都比同 seed 的 base 更差，可能是更大网络初始策略或更新尺度更不稳定，但需要至少 3-5 个 episode 才能判断。
3. `higher_entropy` 的 DSO loss 约为 `-4.141086`，明显大于 base/lower_lr 的 `-1.380362`，符合更强 entropy 设置会改变 actor objective 尺度的预期，需要观察后续是否带来更好探索。
4. `lower_lr` 的 dispatch loss 在 seed 9403 出现 `-0.577246`，明显比其他 case 大，需继续观察是否是随机 batch 波动还是 learning-rate 与 advantage 尺度耦合导致。

---

## 7. 下一步建议

短期：

1. 继续让 12 shard 运行到至少 3 个 episode。
2. 暂时不要继续扩容，避免 GPU OOM。
3. 到 episode 3 后比较：
   - episode reward 变化方向；
   - critic loss 是否爆炸；
   - dispatch loss 是否异常放大；
   - private profit trace 是否仍长期负；
   - reward dynamic cards 中 dispatch / DSO 分项是否有明显失衡。

中期：

如果目标是尽快得到一套主模型，应停止把资源继续投入更多独立 shard，转向实现共享权重并行 rollout。

当前独立 shard 只建议作为短期调参侦察使用，不建议把它作为最终主训练加速方案。
