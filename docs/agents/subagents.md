# DSO 改造子代理定义

Updated: 2026-05-28 Asia/Shanghai

本文件记录本仓库在 `sensitivity_attention_v1` 改造中使用或模拟的子代理职责。
如果工具环境支持真实 subagent，应优先真实调用；如果不支持，则按同一职责顺序执行并写入 handoff。

| 子代理 | 中文功能名 | 职责 |
|---|---|---|
| repo-cartographer | 仓库地图代理 | 定位 DSO envelope、FR/DOE、actor observation、reward、trainer、config、tests，维护 `repo_map.md` |
| schema-architect | 结构化数据架构代理 | 设计 ActionUnit、NetworkObject、StructuredDSOObservation，防止重复 schema、`zone_id` 和 `reliability` 泄漏 |
| sensitivity-engineer | 潮流敏感度代理 | 实现 finite-difference sensitivity、cache、active slice、P/Q channel mask |
| dso-model-engineer | DSO 注意力模型代理 | 实现 bipartite attention actor、edge encoder、self-attention、forward shape tests |
| safe-decoder-engineer | 安全解码代理 | 实现 safe envelope decoder，保证 preferred range 不越过 FR/DOE hard bounds |
| training-stability-engineer | 训练稳定性代理 | 接入 HAPPO/MAPPO 稳定项、warm start、residual schedule、NaN guard 和 loss logging |
| baseline-keeper | 基线守护代理 | 保留 rule_v0、legacy flat observation、legacy MLP actor、旧 trainer configs |
| privacy-auditor | 隐私审计代理 | 审计 DSO actor observation，禁止私有成本、舒适偏好、私有 SOC、oracle 字段泄漏 |
| experiment-harness-engineer | 实验脚手架代理 | 增加 configs、smoke scripts、short training sanity、metrics schema、seed/config hash logging |
| docs-and-memory-maintainer | 文档记忆代理 | 更新 AGENTS、memory、architecture、experiment log、handoff、known failures |

## 当前真实子代理调用记录

### 2026-05-28 Cicero

- 类型：只读仓库扫描。
- 输出：已返回 DSO envelope、actor、FR/DOE、sensitivity、test 文件位置。
- 合并位置：
  - `docs/dso_sensitivity_attention_upgrade_report.md`
  - `docs/agents/repo_map.md`

## 调用边界

- 子代理不能输出或读取密钥、token、密码、私钥。
- 子代理不能回滚用户已有改动。
- 子代理结论必须由主代理用本地文件和测试二次核验。
