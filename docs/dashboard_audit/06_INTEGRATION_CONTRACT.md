# 06 Integration Contract

## 结论

推荐策略 C：混合模式。

训练进程只接入轻量 logger/hook/adapter，默认写稳定 schema 的旁路日志；dashboard 可以作为独立 CLI 读取日志，也可以在本地 debug 时由训练入口可选启动 127.0.0.1 Web 服务。该策略满足用户希望“训练启动时本地启动 Web 服务”的方向，同时避免把长训练、CUDA、多进程 rollout 和 Web server 绑死在同一个阻塞主线程。

## 证据

- 现有 `examples/08_run_dashboard.py` 默认本地 `127.0.0.1:8050`，证明本地服务模式已有先例。
- 现有 Dash 只读 CSV，不是实时 event bus。
- `paper_training.py` 已有 `experiment_progress.jsonl`、CSV/TensorBoard artifacts，可作为低侵入扩展点。
- HAPPO shared rollout 和 paper-long CUDA guard 说明长训练需要资源隔离。
- `pyproject.toml` 已将 visualization 依赖放在 optional `viz`，说明 dashboard 依赖应继续可选化。

## 相关文件路径

- `src/vpp_dso_sim/experiments/paper_training.py`
- `src/vpp_dso_sim/envs/multi_agent_env.py`
- `src/vpp_dso_sim/simulation/simulator.py`
- `src/vpp_dso_sim/learning/advanced_marl.py`
- `src/vpp_dso_sim/learning/matd3.py`
- `src/vpp_dso_sim/learning/hatrpo.py`
- `src/vpp_dso_sim/dashboard/app.py`
- `examples/08_run_dashboard.py`
- `pyproject.toml`

## 相关类/函数/变量

- `run_paper_training_experiment`
- `_print_progress`
- `_append_progress_event`
- `_write_tensorboard_scalars`
- `MultiAgentVPPDSOEnv.step`
- `Simulator.records`
- `train_happo`
- `train_hasac`
- `train_matd3`
- `train_hatrpo`
- `progress_callback`
- `happo_shared_rollout_enabled`

## 推荐架构

```text
Training process
  -> lifecycle hooks
  -> EnvironmentAdapter / AlgorithmAdapter / VariableDictionary
  -> async queue
  -> single writer sink
       JSONL now
       optional SQLite/DuckDB/Parquet later

Dashboard service
  -> independent CLI by default
  -> optional local auto-start on 127.0.0.1
  -> reads run log directory
  -> never mutates env/algorithm/reward
```

## 线程/进程模型

- 默认：训练进程写日志，dashboard 独立进程读取。
- debug 可选：训练入口启动一个后台 Web 服务进程，仍通过日志目录通信。
- 不建议：在训练主线程内直接运行阻塞 Web server。
- 多进程 rollout：worker 不直接写共享 dashboard DB；由主进程收集或每 worker 写独立 fragment 后合并。

## TrainingLifecycleContract

```python
class TrainingLifecycleContract:
    def on_train_start(self, context: dict) -> None: ...
    def on_env_reset(self, context: dict, observations: object, infos: dict) -> None: ...
    def on_env_step(self, context: dict, transition: dict) -> None: ...
    def on_reward_computed(self, context: dict, reward_components: dict) -> None: ...
    def on_cost_computed(self, context: dict, cost_components: dict) -> None: ...
    def on_loss_computed(self, context: dict, loss_components: dict) -> None: ...
    def on_episode_end(self, context: dict, episode_metrics: dict) -> None: ...
    def on_epoch_end(self, context: dict, epoch_metrics: dict) -> None: ...
    def on_eval_end(self, context: dict, eval_metrics: dict) -> None: ...
    def on_checkpoint_saved(self, context: dict, checkpoint: dict) -> None: ...
    def on_train_error(self, context: dict, error: BaseException) -> None: ...
    def on_train_end(self, context: dict, summary: dict) -> None: ...
```

Implementation rule：所有方法必须 no-op safe，内部捕获异常并降级 warning。Hook 不返回任何会影响训练决策的值。

## MetricSchemaContract

必须字段：

| field | type | nullable | 说明 |
|---|---|---|---|
| `run_id` | string | no | 唯一 run |
| `epoch_id` | int/string | yes | PPO/value optimizer epoch，不强行等同 campaign epoch |
| `episode_id` | int | yes | env reset 到 done/truncated |
| `batch_id` | int/string | yes | replay/minibatch 或 rollout batch |
| `gradient_step` | int | yes | adapter 合成或原生 update counter |
| `global_env_step` | int | yes | adapter 合成，shared rollout 要带 env_id |
| `env_id` | string | yes | 单进程可为 `env_0` |
| `vpp_id` | string | yes | VPP 实体 |
| `agent_id` | string | yes | env agent |
| `policy_id` | string | yes | 策略/actor id |
| `date` | string | yes | 当前未发现真实 date |
| `time_index` | int | yes | scenario step |
| `timestamp` | string | yes | 当前未发现真实 timestamp |
| `metric_group` | string | no | dataset/reward/cost/loss/grid/action |
| `metric_name` | string | no | 指标名 |
| `value` | float/string/bool | no | 指标值 |
| `unit` | string | yes | 单位 |
| `formula_latex` | string | yes | 公式 |
| `description` | string | yes | 物理意义 |

## EnvironmentAdapterContract

从现有 env 中只读提取：

- `date`：默认 `None`；若 profile metadata 有真实起始日期，再计算。
- `time_index`：`env.current_step` 或 `result["step"]`。
- `timestamp`：默认 `None`；不要从当前日期推断。
- `vpp_id`：从 scenario.vpps 或 agent_id 解析。
- `observation`：raw dict 或 encoded vector，标记 `normalized/encoded`。
- `action`：保存 raw action、decoded action、projected action、landed action。
- `reward`：从 `rewards` dict 和 `reward_components` 读取。
- `cost`：从 DSO components、settlement、violation penalties 读取。
- `done/terminated/truncated`：按 env 返回值原样记录。
- `info`：只读记录 `action_validation`、`critic_global_state`、`violations` 等。
- `dataset physical values`：优先从 `Simulator.records` 读取 `profile_state`、grid state、vpp_power、der_dispatch、SOC、settlement。

## AlgorithmAdapterContract

从 algorithm/trainer 中只读提取：

- `actor_loss`
- `critic_loss`
- `entropy_loss`
- `value_loss`
- `q_loss`
- `total_loss`
- `optimizer_name`
- `network_name`
- `policy_id`
- `gradient_step`
- `learning_rate`
- `gradient_norm` 可选

算法映射：

- HAPPO：role update rows + critic update rows。
- HATRPO：value loss + trust-region KL/line-search metrics。
- MATD3：critic/actor update metrics、Q head metrics、replay size。
- HASAC：critic/actor/alpha update metrics。
- DeepRL legacy：policy/value/loss summary，标记 legacy。

## VariableDictionaryContract

字段：

```text
name
display_name
symbol
unit
group
physical_meaning
formula_latex
min_value
max_value
source
notes
```

维护原则：

- 变量字典独立于前端组件。
- 每个 metric row 引用 `metric_name`，可 join variable dictionary。
- 物理量和 normalized feature 分开定义。
- 未确认单位使用 `unit="unknown"` 或 `unit="currency/MWh?"`，不要硬编码。

## 数据流

```text
env/algorithm event
  -> adapter normalizes to MetricSchemaContract rows
  -> queue.put_nowait(batch)
  -> writer thread/process flushes by interval or row count
  -> manifest marks schema_version and completed/failed state
  -> dashboard reads tail/aggregate APIs
```

## 异常隔离

- Logger 初始化失败：训练继续，输出 warning。
- Queue 满：丢弃低优先级高频 metrics，保留 train/error/checkpoint events。
- Writer 失败：关闭 logger，训练继续。
- Dashboard 服务失败：不影响训练。
- Schema validation 失败：记录到 `logger_errors.jsonl`，不抛到训练主循环。

## 风险

- High：如果在 hook 中访问 GPU tensor 并同步 `.cpu()`，可能拖慢训练；adapter 应只接收已经标量化的 metrics。
- Medium：JSONL 对超大长训可读性好但聚合慢；后续可选 DuckDB/Parquet。
- Medium：同进程 Web 服务仍可能与 multiprocessing/spawn 冲突；默认应独立进程读取。

## 建议

- 首批实现只做 JSONL + manifest + variable dictionary，不引入 DB。
- 可选依赖分组命名建议：`dashboard` 或 `realtime-dashboard`。
- Web 服务默认只绑定 `127.0.0.1`。
- 不把 existing Dash `src/vpp_dso_sim/dashboard/app.py` 改为实时服务；新模块并存。

## 待用户确认项

- 是否接受 JSONL 作为第一版日志格式，DuckDB/Parquet 作为后续优化。
- 是否要求训练启动自动打开 Web 服务；若是，默认子进程还是只打印独立 dashboard CLI 命令。
- 是否需要远程访问；若需要，隐私脱敏/鉴权必须前置。
