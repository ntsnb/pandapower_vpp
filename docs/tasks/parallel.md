我希望你帮助我把当前项目中的 HAPPO 训练模式，从“单环境顺序 rollout / 多 shard 独立训练”扩展为“共享权重同步并行 rollout”。

请先阅读代码仓，不要假设具体文件路径、类名、函数名或配置名。所有命名、路径、入口、数据结构都以真实代码为准。如果下面描述中的术语和代码仓已有术语不一致，请优先使用代码仓现有术语，只保持算法语义一致。

背景目标：
当前项目可以通过多个 shard 同时训练不同 seed / hparam case，例如 base、lower_lr、higher_entropy、larger_network。这种方式适合快速比较配置，但它训练的是多套独立模型。现在我想新增一种模式：多个环境 worker 同时为同一套 HAPPO 模型采样，然后合并数据，对同一套 actor / critic 统一更新，从而加速单个主模型训练。

请保留现有训练模式。新增功能默认关闭，不要破坏原有 shard、多 seed、多 hparam case、reward、环境、网络结构和已有实验 preset。

需要实现的目标分两部分。

第一部分：短跑 shard probe
保留或补充一种方式，让当前多个 shard 只短跑 1 到 3 个 episode，用于比较 base、lower_lr、higher_entropy、larger_network 哪个更稳定。请优先复用现有训练入口和现有 shard 机制，不要重复造 launcher。需要能覆盖最大训练 episode 数或等价训练长度。

请提供一个轻量 summary 能力，汇总短跑结果，帮助选择主配置。summary 至少尽量包含：
- seed
- hparam case
- final reward 或 mean reward
- reward 波动
- critic loss / actor loss，如果已有记录
- entropy / KL / clip fraction，如果已有记录
- safety violation / private profit / projection gap，如果已有记录
- 是否崩溃、是否 NaN/Inf、是否未完成

如果某些指标代码里没有现成记录，可以跳过，不要为了 summary 大改训练逻辑。

第二部分：HAPPO 共享权重同步并行 rollout
新增一种 HAPPO 训练模式，核心语义如下：

1. 每一轮 update 开始时，固定当前主模型参数，作为本轮 rollout 的行为策略快照。
2. 所有 worker 必须使用同一个行为策略快照采样。
3. rollout 阶段不能让 worker 各自更新模型。
4. 所有 worker 采样得到的 trajectory / fragment 合并成一个 batch。
5. 在合并 batch 上计算 return、GAE、advantage。
6. 使用合并 batch，按现有 HAPPO 的 agent 顺序更新 actor。
7. 使用合并 batch 更新 centralized critic。
8. 更新完成后，再进入下一轮采样。

请注意：这里要实现的是同步 on-policy parallel rollout，不是异步 worker 各自训练，也不是多个 shard 各自训练。worker 之间不能各自持有不同版本的训练中策略并局部 update。所有 worker 的数据必须属于同一个 policy version。

新增配置建议：
请根据项目现有配置风格增加等价能力，不要求使用以下精确名字：
- worker 数，默认 1
- rollout fragment length，默认保持原行为
- 是否启用 shared rollout，默认 false 或由 worker 数大于 1 触发
- 后端类型，可先支持 serial/vectorized，之后再支持 multiprocessing
- 最大 episode 数或训练步数 override，用于短跑 probe

默认建议：
- 先支持 4 到 6 个 worker
- fragment length 先支持 96 或 168
- 不要默认直接使用 12 个 worker
- 如果 multiprocessing 风险较高，先实现 serial/vectorized 版本，保证算法正确；再扩展到 multiprocessing

关键正确性要求：
请重点保证以下内容，不要把它们当成普通日志字段：

1. old log probability
每条样本必须保存采样时行为策略对实际采样动作的 log probability。训练时用当前策略重新评估同一个旧动作，计算 ratio。不要在 update 阶段用当前策略伪造 old log probability。

2. worker/time/agent 维度
rollout buffer 必须能区分 worker、时间步和 agent。具体 shape 由你根据现有代码设计，但不要把不同 worker 的时间轴错误拼接。

3. GAE
GAE 必须沿每个 worker 自己的时间轴计算。不同 worker 的 trajectory 不能前后相接。fragment 结尾如果不是环境真实终止，应使用 critic bootstrap。真实 terminal 不 bootstrap。

4. done / truncated / fragment cut
请检查现有环境对 done、terminated、truncated 或 time-limit 的定义。人为 rollout fragment 截断不能简单等同于真实 episode 结束。需要正确处理 bootstrap。

5. advantage normalization
优先在合并后的 batch 上做 normalization，而不是每个 worker 单独 normalize，除非现有 HAPPO 实现有明确不同语义。若现有代码有自己的标准做法，请保持语义并说明。

6. centralized critic
critic 必须使用每条样本对应的 global state / centralized state。不要把多个 worker 的状态平均后给 critic，也不要只用某一个 worker 的状态。

7. HAPPO agent sequential update
必须保留现有 HAPPO 的 agent 顺序更新逻辑。不要因为并行 rollout 把多个 agent 改成完全独立同时更新。并行的是环境采样，不是 agent update 逻辑。

8. action mask / action projection / reward components
如果当前项目中已有 action mask、动作投影、安全修正、reward 分解或 private profit 记录，请保证 shared rollout 下不会丢失或错位。具体保存方式可以按现有结构设计。

9. policy version
建议给每轮 rollout 数据记录 policy version 或等价标识。update 前检查同一个 merged batch 是否来自同一个行为策略快照。若混入不同版本，应报错或拒绝 update。

请实现必要的测试或 smoke check：
- 原有单 worker HAPPO 路径不受影响。
- worker 数为 1 时，新路径与原逻辑等价或非常接近。
- 多 worker shared rollout 能完成至少一次 rollout + update。
- update 刚开始时，当前策略和行为策略相同，ratio 均值应接近 1。
- GAE 不跨 worker。
- fragment cut 使用 bootstrap，真实 terminal 不使用 bootstrap。
- merged batch 中所有样本来自同一个 policy version。
- actor 和 critic 都完成一次 optimizer step。
- 不出现 NaN/Inf。
- 如果实现 multiprocessing，worker 能正常启动、关闭，异常时不残留进程。

实现顺序建议：
1. 先定位现有 HAPPO rollout、buffer、GAE、actor update、critic update 和训练入口。
2. 增加配置项和 CLI / config 接入，默认关闭。
3. 实现短跑 probe 的 episode override 和 summary。
4. 实现 serial/vectorized shared rollout。
5. 接入现有 HAPPO update，不改变 agent sequential update 语义。
6. 增加关键 debug metrics，例如 ratio、KL、clip fraction、advantage mean/std、bootstrap value。
7. 跑 smoke tests。
8. 如果 serial 版本稳定，再实现或预留 multiprocessing backend。

请完成后给出工程报告：
- 你实际修改了哪些文件
- 新增了哪些配置或命令行参数
- 如何运行 1 到 3 episode 的 shard probe
- 如何查看 probe summary 并选择主配置
- 如何启动 HAPPO shared rollout，先用 4 workers + 96 steps
- 哪些测试已经通过
- 哪些点仍需要人工确认

请不要过度重构。目标是最小侵入地增加 HAPPO 共享权重同步 rollout，加速单个主模型训练，同时保留当前独立 shard 方案用于短跑筛选配置。
