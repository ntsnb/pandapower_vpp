# 并行训练方案对比报告：独立 Shard 实验并行 vs 共享权重并行 Rollout

生成时间：2026-06-12

项目路径：`/mnt/sda/home/niutiansen/panda power/pandapower-vpp-dso-sim`

当前背景：项目已经启动过 `paper_long_sensitivity_v1` 的并行 shard 版本。该版本当前采用的是“多个独立实验并行”，不是“同一套神经网络权重并行采样后统一更新”。

---

## 1. 你的想法的正式化表述

你希望优先更快训练出一套最终可用的强化学习策略，而不是同时训练多套不同 seed 或不同超参数的策略。

更正式地说，你想要的是：

> 使用同一套 DSO actor、dispatch actor、portfolio actor 和 centralized critic 参数，在多个并行环境中同时采样。每个环境 worker 使用当前相同策略与不同场景、不同随机种子或不同时间片交互，得到多条 rollout 轨迹。然后把这些轨迹合并为一个更大的 batch，用于对同一套 actor/critic 执行一次或若干次统一更新。

可以写成：

```text
当前共享参数：
θ_actor, θ_critic

并行环境：
Env_1, Env_2, ..., Env_N

每个环境使用同一套旧策略：
π_{θ_old}

并行采样：
τ_i = rollout(π_{θ_old}, Env_i), i = 1...N

合并 rollout：
B = τ_1 ∪ τ_2 ∪ ... ∪ τ_N

统一计算：
advantage(B), return(B), value_target(B)

统一更新：
θ_new = Update(θ_old, B)
```

这可以称为：

- 共享权重并行 rollout
- vectorized rollout
- parallel environment sampling
- synchronous on-policy parallel training
- 对 HAPPO/HATRPO 来说，可以理解为“多环境同步采样 + 单模型集中更新”

这个方案最后主要产出一套主模型权重，而不是每个 shard 各自训练一套权重。

---

## 2. 当前我启动的 shard 方案是什么

当前 `sharded_v2` 采用的是：

```text
seed_9401_base             -> 一套独立权重
seed_9401_lower_lr         -> 一套独立权重
seed_9401_higher_entropy   -> 一套独立权重
seed_9401_larger_network   -> 一套独立权重

seed_9402_base             -> 一套独立权重
...
seed_9403_larger_network   -> 一套独立权重
```

每个 shard 是完整独立的训练任务：

```text
θ_1 单独初始化、单独采样、单独更新、单独保存 checkpoint
θ_2 单独初始化、单独采样、单独更新、单独保存 checkpoint
...
θ_N 单独初始化、单独采样、单独更新、单独保存 checkpoint
```

因此，当前 shard 并行的作用是：

- 同时跑多个 seed；
- 同时跑多个超参数；
- 更快完成 paper-long 实验矩阵；
- 更快比较哪个配置稳定；
- 更快获得多 seed 统计；
- 但不会让“某一套模型”本身训练得更快。

这类方案可以叫：

- 实验矩阵并行
- independent experiment sharding
- multi-seed / multi-hparam sweep
- 多实验并行，不是单模型并行训练

---

## 3. 两种方案的核心区别

| 维度 | 当前独立 shard 方案 | 你想要的共享权重并行 rollout |
|---|---|---|
| 训练目标 | 同时训练多套模型 | 更快训练一套模型 |
| 权重数量 | 每个 shard 一套权重 | 所有 worker 共享一套权重 |
| 参数是否同步 | 不同步 | 同步 |
| rollout 来源 | 每套模型自己的环境 | 多个环境给同一套模型采样 |
| batch 构成 | 每个实验内部独立 batch | 多个环境轨迹合并为一个 batch |
| 更新方式 | 每个 shard 单独 update | 合并后统一 update |
| 输出 checkpoint | 多套 checkpoint | 一套主 checkpoint |
| 适合目的 | 超参数搜索、多 seed 统计、论文稳健性 | 缩短单模型训练时间、提高采样吞吐 |
| 实现难度 | 低，当前 CLI 已支持 | 中到高，需要改训练循环 |
| 对现有代码侵入 | 小 | 较大 |
| 是否马上可用 | 已可用 | 需要实现和测试 |

---

## 4. 当前独立 shard 方案的优点

### 4.1 工程风险低

它不需要重写 HAPPO/HATRPO 的核心训练循环。每个 shard 调用现有 CLI：

```bash
python examples/17_paper_training_experiment.py \
  --preset paper_long_sensitivity_v1 \
  --algorithms happo \
  --seeds 9401 \
  --hparam-cases base
```

现有代码已经支持 `--seeds` 和 `--hparam-cases`，因此这个方案主要是调度层面的并行，不涉及算法正确性风险。

### 4.2 很适合论文级实验

论文通常需要：

- 多 seed；
- 多场景；
- 多超参数对照；
- 多 checkpoint 评估；
- 不同算法比较；
- 统计均值、方差、置信区间。

独立 shard 正好适合快速完成这些实验矩阵。

### 4.3 失败隔离好

如果某一个 seed 或某一个 hparam case 崩溃，只影响那个 shard。其他 shard 仍然继续运行。

### 4.4 方便调参筛选

比如 `base`、`lower_lr`、`higher_entropy`、`larger_network` 同时跑，可以较早观察：

- 哪个 reward 曲线更稳定；
- 哪个 critic loss 更低；
- 哪个 actor loss 没有崩；
- 哪个 safety violation 更少；
- 哪个 dispatch private profit 更合理。

---

## 5. 当前独立 shard 方案的缺点

### 5.1 不会加速单个模型的收敛

这是最关键的问题。

12 个 shard 同时跑，只是同时训练 12 套模型。它不会把 12 个环境的经验合并起来帮助同一个 HAPPO 模型更新。

也就是说：

```text
实验矩阵完成速度提高了
但单个模型每个 episode 的采样速度基本没有提高
```

当前观察也印证这一点：每个 shard 仍然大约 `6.6 - 7.3 秒/step`。

### 5.2 资源会被分散

GPU 和 CPU 被多套模型共享，而不是集中服务于一套模型。因此如果你只关心最终一套最强策略，这种方案会显得“分散火力”。

### 5.3 结果数量多，分析成本高

它会产生多套：

- checkpoint；
- reward 动态卡片；
- loss 曲线；
- evaluation summary；
- tensorboard；
- progress csv。

后续需要聚合、筛选、比较。

---

## 6. 共享权重并行 rollout 方案的优点

### 6.1 更符合你的目标

你的目标是：

```text
优先更快训练出一套模型
```

共享权重并行 rollout 正是为这个目标服务。

它不是训练 12 套模型，而是让 12 个环境同时给一套模型提供样本。

### 6.2 可以提高单次 update 的样本量

假设单环境一次 rollout 是：

```text
672 steps × 1 env
```

如果使用 8 个并行环境，则一次同步 rollout 可以变成：

```text
672 steps × 8 envs
```

或者为了保持 batch 总量不变，也可以改成：

```text
84 steps × 8 envs = 672 total samples
```

这两种设计不同：

1. 保持每个 worker 672 steps：更大 batch，更稳定，但每轮 update 前等待时间仍长。
2. 每个 worker 缩短到 84 或 96 steps：更频繁 update，单模型推进更快，但需要确认 HAPPO 的 advantage/GAE 处理仍合理。

### 6.3 可以改善样本多样性

多个环境 worker 可以使用：

- 不同随机种子；
- 不同负荷扰动；
- 不同 PV 云遮挡；
- 不同 EVCS 需求；
- 不同初始 SoC；
- 不同 VPP 组合扰动；
- 不同 train_mixed 子场景。

这样同一套策略每次更新能看到更丰富的状态分布。

### 6.4 对 GPU 更友好

当前 GPU 利用率低，是因为大部分时间花在 CPU 环境 step 上。并行 rollout 可以让 CPU 同时生成更多样本，然后在 update 时给 GPU 更大的 batch。

GPU 利用率不一定持续很高，但 update 阶段会更有效率。

---

## 7. 共享权重并行 rollout 方案的缺点和风险

### 7.1 需要改训练循环

现有 HAPPO 训练大概率是：

```text
单环境 reset
for episode:
    for step in horizon:
        actor act
        env.step
        store transition
    update actor/critic
```

共享 rollout 需要改成：

```text
构造 N 个 env worker
同步广播 θ_old
for rollout_step:
    每个 env worker 使用同一策略采样
    收集 N 个 transition
合并到 rollout buffer
统一计算 GAE/return
统一更新 actor/critic
```

这会影响：

- rollout buffer 数据结构；
- observation/action/reward 批处理；
- done/truncated 处理；
- per-agent advantage；
- centralized critic 输入；
- reward normalization；
- observation normalization；
- checkpoint 记录；
- dynamic reward report；
- private profit trace；
- tensorboard 标量；
- evaluation 入口。

### 7.2 HAPPO 是 on-policy，不能随便异步更新

HAPPO/PPO/TRPO 类算法要求样本来自当前旧策略 `π_{θ_old}`。

正确做法是：

```text
所有 worker 在同一轮 rollout 中使用同一个冻结的 θ_old
rollout 收集完后统一 update
update 完成后再把 θ_new 广播给所有 worker
```

不应让每个 worker 各自边采样边更新，否则会变成混乱的异步 on-policy/off-policy 混合，可能破坏 HAPPO 的理论假设。

### 7.3 多 worker 的 episode 长度和 GAE 处理复杂

如果每个 worker 都跑完整 672-step episode，逻辑简单但 update 周期仍很长。

如果每个 worker 跑短片段，比如每次 96 steps，就必须处理：

- bootstrap value；
- truncated mask；
- GAE 跨片段截断；
- time feature；
- SoC 等跨时序状态；
- 终端 SoC 或长期成本的 credit assignment。

这个项目有储能 SoC、EVCS、PV、价格和 672-step 长时序，因此不能粗暴切短而不处理 bootstrap。

### 7.4 报告和审计要改

当前每个 episode 生成：

- reward dynamic cards；
- dispatch private profit trace；
- settlement audit；
- train progress；
- loss metrics。

共享 rollout 后会出现：

```text
env_worker_0 episode 1
env_worker_1 episode 1
...
env_worker_N episode 1
```

报告必须明确区分：

- worker id；
- scenario seed；
- policy update index；
- physical episode index；
- global environment step；
- per-worker episode reward；
- aggregated batch reward。

否则后续你会看不清“一个 episode 的 reward”到底来自哪个环境。

---

## 8. 对 HAPPO/HATRPO 的具体影响

### 8.1 HAPPO

HAPPO 比较适合做同步并行 rollout。

推荐结构：

```text
N 个并行 env
共享 actor/critic
每个 update 收集 N 条 rollout
合并后计算 advantage
按 agent 顺序做 HAPPO sequential update
```

关键要求：

- 所有 worker 用同一个 `θ_old`；
- advantage normalization 要在合并 batch 上做，而不是每个 worker 单独做；
- centralized critic 要看到每条样本对应的 global state；
- agent update 顺序要保持 HAPPO 原实现逻辑；
- log_prob_old 必须随样本保存；
- action mask / projection gap / reward trace 必须按 worker 保存。

### 8.2 HATRPO

HATRPO 也可以做共享 rollout，但更复杂。

HATRPO 的 trust-region / conjugate-gradient / Fisher-vector product 对 batch 结构更敏感。并行 rollout 的好处是 Fisher 估计更稳定，但实现时必须确保：

- KL 在合并 batch 上统计；
- surrogate loss 在合并 batch 上统计；
- Fisher-vector product 支持合并后的多 worker 样本；
- 不引入 CPU flash attention 二阶梯度问题。

之前 dispatch set-attention 已经避免使用 PyTorch flash attention，因此 HATRPO 路径更安全，但仍需要专门测试。

---

## 9. 两种方案的速度收益对比

### 当前独立 shard 方案

如果 12 个 shard 同时跑：

```text
单个模型速度：约 7 秒/step
总实验吞吐：约 12 / 7 = 1.7 environment-steps per second
```

它提升的是总实验矩阵吞吐。

### 共享权重并行 rollout

理想情况下：

```text
单个模型一次 update 可获得 N 倍样本
```

但 wall-clock 是否显著变快，取决于设计：

#### 方案 A：N 个 worker 都跑完整 672 steps

```text
优点：batch 大，估计稳定
缺点：一次 update 前仍要等完整 episode，单次 update 等待时间没有明显缩短
```

#### 方案 B：N 个 worker 跑短 rollout，比如 96 steps

```text
优点：更频繁 update，单模型训练推进更快
缺点：需要正确处理 bootstrap、GAE、truncated、长期 SoC
```

我更推荐从方案 B 的保守版本开始：

```text
num_workers = 4
rollout_fragment_steps = 168 或 96
aggregate_batch_steps = num_workers × rollout_fragment_steps
```

先不要一上来 12 worker，因为多 worker 会同时放大日志、显存、进程通信、报告聚合复杂度。

---

## 10. 两种方案是否可以结合

可以，但优先级要明确。

推荐路线：

### 阶段 1：保留当前 12 shard 跑短观察

目的不是得到最终论文结果，而是快速看：

- 哪个 hparam case 最稳定；
- base、lower_lr、higher_entropy、larger_network 谁更适合作为主配置；
- reward 是否继续恶化；
- critic loss 是否爆炸；
- safety violation 是否为 0；
- dispatch private profit 是否仍长期负值。

这相当于用 shard 并行做“调参侦察”。

### 阶段 2：选出一个主配置

例如：

```text
seed: 9401
hparam_case: lower_lr
algorithm: happo
dispatch_actor_encoder_type: set_attention_v1
reward: v3_1_market_safety
```

具体选哪个，要等当前 shard 至少跑出几个 episode 后看曲线。

### 阶段 3：实现共享权重并行 rollout

用选出来的主配置实现：

```text
一套模型
4 个并行环境
短 rollout fragment
合并 batch
统一 HAPPO update
```

### 阶段 4：再做 paper-long

先用共享 rollout 更快训练一个主模型。确认收敛后，再做多 seed 统计。

---

## 11. 如果采用共享权重并行 rollout，需要改哪些代码

初步修改清单：

| 文件 | 修改内容 | 原因 |
|---|---|---|
| `src/vpp_dso_sim/learning/advanced_marl.py` | 为 HAPPO 增加 parallel rollout runner | 当前 HAPPO 是单环境顺序采样 |
| `src/vpp_dso_sim/learning/deep_rl.py` 或相关 rollout buffer 文件 | 增加 worker 维度的 rollout buffer | 保存多个环境的 obs/action/reward/done/log_prob |
| `src/vpp_dso_sim/envs/multi_agent_env.py` | 确认环境可多实例并行安全创建 | 每个 worker 需要独立 env，不共享 pandapower net 可变状态 |
| `src/vpp_dso_sim/experiments/paper_training.py` | 增加 `num_rollout_workers`、`rollout_fragment_steps` 配置 | paper-long 入口要能开启并行 rollout |
| `examples/17_paper_training_experiment.py` | 增加 CLI 参数 | 便于命令行控制 worker 数 |
| `src/vpp_dso_sim/visualization/reward_dynamic_report.py` | 报告增加 worker id/update id | 防止多 worker 报告混淆 |
| `tests/test_hasac_happo.py` 或新增测试 | 测试 parallel rollout 形状、worker 合并、checkpoint | 防止算法 silently wrong |

---

## 12. 最小可行实现建议

不要直接实现复杂分布式框架。建议先做一个保守同步版本：

```text
num_rollout_workers = 4
rollout_fragment_steps = 96 或 168
每个 worker 一个独立 env
所有 worker 在同一进程内顺序或 multiprocessing 并行 step
采样后合并 buffer
统一 update
```

为了降低风险，可以分两步：

### 12.1 第一版：单进程 vectorized 逻辑

同一个 Python 进程里创建多个 env，但先顺序 step：

```text
for t in rollout_fragment_steps:
    for worker in workers:
        action = policy(obs_worker)
        next_obs = env_worker.step(action)
```

这不会显著加速 CPU，但可以先验证 batch 合并和算法逻辑正确。

### 12.2 第二版：multiprocessing 并行 env.step

确认算法逻辑正确后，再把 env step 放到多个进程里：

```text
worker processes:
    receive action
    run env.step(action)
    return transition
```

这才是真正加速 pandapower 环境采样的版本。

---

## 13. 决策建议

如果你的首要目标是：

```text
尽快得到论文多 seed、多超参数结果
```

则继续当前独立 shard 方案。

如果你的首要目标是：

```text
尽快把一套 HAPPO/dispatch set-attention 模型训练到收敛
```

则应该转向共享权重并行 rollout。

基于你刚才的表述：

> “我不想训练出多份不同参数设置的结果，我想优先更快地练出一套实验”

我建议：

```text
暂停继续扩展独立 shard；
保留当前 12 shard 只作为短时调参观察；
尽快实现共享权重并行 rollout；
选一个主 hparam case 后集中训练一套主模型。
```

当前 12 shard 不一定要马上全部杀掉。更稳妥的做法是让它们至少完成 1-3 个 episode，得到初步曲线，用来选择主配置。然后停止多余 shard，把资源集中给共享权重并行 rollout 主实验。

---

## 14. 一句话总结

当前 shard 并行是：

```text
多套模型同时训练，用来更快完成实验矩阵。
```

你想要的是：

```text
一套模型用多个环境同时采样，把多环境数据合成一个 batch 后统一更新。
```

两者都叫“并行”，但算法含义完全不同。若目标是更快训练出一套主模型，应优先实现共享权重并行 rollout，而不是继续增加独立 shard 数量。
