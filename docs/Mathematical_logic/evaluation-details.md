# 数理逻辑测评脚本细节说明

本文档补充说明 `utils/evaluate_policy.py` 的 policy 接口、命令行参数、JSON 输出字段和观测变体细节。精简版测评规则见 [数理逻辑任务测评说明](evaluation.md)。

## 一、评分口径

策略性能的核心指标是任务完成率：

```text
success_rate = 成功完成任务的 episode 数 / 总 episode 数
```

其中“成功完成任务”以环境返回的任务完成信号为准。脚本会检查：

- `info["game"]["world_completed"] == True`
- 或 `info["terminal_reason"] == "world_completed"`
- 或 reward 终止信息中的 `terminated_reason == "world_completed"`

因此，正确率不是按单步动作是否正确计算，而是按一个 episode 是否最终完成该关任务计算。

除成功率外，脚本还会统计进度信息，用于观察未完全通关的 agent 已完成哪些阶段性目标。进度统计包括：

- 物品搜集：如 `key_collected`、`gold_collected`、`item_collected`。
- 宝箱与交互：如 `chest_opened`、`button_pressed`、`door_opened`。
- 战斗与安全：如 `monster_killed`、`trap_triggered`、`agent_dead`。
- 地图推进：如 `room_changed`、`exit_reached`、`environment_completed`。

这些进度指标不会替代成功率，但可以作为报告中分析策略行为、阶段性完成度和失败原因的依据。

## 二、测评脚本用途

测评脚本用于在五个数理逻辑任务上黑盒运行提交的 policy，并统计每个任务、每个测评阶段的成功率、平均步数、平均奖励和关键进展。

最小运行命令：

```bash
python utils/evaluate_policy.py --policy docs/Mathematical_logic/examples/agent.py
```

指定任务和 episode 数：

```bash
python utils/evaluate_policy.py \
  --policy docs/Mathematical_logic/examples/agent.py \
  --tasks mathematical_logic/task_1 mathematical_logic/task_3 \
  --num-envs 10
```

输出 JSON 结果：

```bash
python utils/evaluate_policy.py \
  --policy docs/Mathematical_logic/examples/agent.py \
  --num-envs 10 \
  --json-out results/evaluation.json
```

脚本默认使用：

- `observation_mode="pixels"`
- Gymnasium 风格的 `reset()` / `step()` 接口
- 五个默认任务：`mathematical_logic/task_1` 到 `mathematical_logic/task_5`

## 三、提交 Agent 接口

`--policy` 可以指向 Python 文件，也可以指向可 import 的模块。脚本会按以下顺序寻找可调用对象：

1. `make_policy()`：调用后返回 policy 对象。
2. `Policy`：实例化该类。
3. `policy`：模块级 policy 对象。
4. `act`：模块级动作函数。

推荐格式：

```python
class Policy:
    def reset(self, seed=None, task_id=None):
        pass

    def act(self, obs, info):
        return 0


def make_policy():
    return Policy()
```

`act()` 的返回值必须能转换为一个整数动作。动作编号见 [task.md](task.md) 中的动作空间说明。

脚本会优先调用：

```python
policy.act(obs, info)
```

如果函数只接受一个参数，则回退为：

```python
policy.act(obs)
```

注意：鲁棒性测评中的颜色和重画变体只修改传给 policy 的 `obs`，不会修改 `info`。`spatial` 阶段会使用临时生成的扰动地图，但任务奖励和完成条件仍使用原 task。

## 四、命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--policy` | 可选 | 所有任务共用的 policy 文件或模块路径，可带 `:attribute` |
| `--task-policy` | 可选 | 单个任务的 policy，格式为 `TASK_ID=POLICY_SPEC`，可重复传入 |
| `--tasks` | 五个数理逻辑任务 | 要测评的任务 ID 列表 |
| `--num-envs` | `100` | 普通测评时表示每个任务、每种观测变体的 episode 数；启用 `--robustness-suite` 时表示每个任务的总 episode 数 |
| `--seed` | `0` | 起始随机种子，第 `i` 个 episode 使用 `seed + i` |
| `--max-steps` | 任务默认值 | 覆盖任务最大步数 |
| `--action-repeat` | 任务默认值 | 覆盖环境的 `action_repeat` |
| `--render-mode` | `None` | 可设为 `rgb_array` |
| `--obs-variants` | `default` | 传给 policy 前应用的像素观测变体 |
| `--robustness-suite` | 关闭 | 启用固定比例鲁棒性测评套件 |
| `--json-out` | `None` | 将详细结果写入 JSON 文件 |

示例：

```bash
python utils/evaluate_policy.py \
  --policy submissions/student_policy.py \
  --tasks mathematical_logic/task_1 mathematical_logic/task_2 \
  --num-envs 20 \
  --seed 100 \
  --action-repeat 4 \
  --json-out results/student_policy_eval.json
```

`--action-repeat` 用于临时覆盖环境每次外部动作重复执行的内部步数。未传入时，脚本使用任务配置文件中的默认值；传入时，仅影响本次测评，不修改环境配置文件。

如果没有提供统一的 `--policy`，则必须通过 `--task-policy` 为每个被测任务都指定 agent。若同时提供 `--policy` 和 `--task-policy`，对应 task 会优先使用 `--task-policy` 指定的 agent，其余 task 使用统一 agent。

## 五、结果含义

每个 episode 会打印一行：

```text
mathematical_logic/task_1 stage=original obs_variant=default map_variant=default seed=0 success=True steps=243 reward=9.120
```

字段含义：

| 字段 | 含义 |
|---|---|
| `task_id` | 当前任务 |
| `stage` | 当前测评阶段 |
| `obs_variant` | 当前观测变体 |
| `map_variant` | 当前地图变体，`default` 表示原始地图，`spatial_a/b/c` 表示坐标扰动地图 |
| `seed` | 当前 episode 的随机种子 |
| `success` | 是否完成任务目标 |
| `steps` | episode 执行步数 |
| `reward` | episode 累计奖励 |

汇总结果按 `(task_id, stage)` 分组，避免不同测评阶段的成功率混在一起：

```text
mathematical_logic/task_1 [redraw]
  episodes:     10
  success_rate: 0.800
  avg_steps:    231.4
  avg_reward:   7.520
  variants:     {'redraw_geometric': 5, 'redraw_symbols': 5}
  map_variants: {'default': 10}
```

`success_rate` 是主要性能指标，计算依据是该阶段内 episode 是否完成任务。脚本还会输出 `variants` 和 `map_variants`，分别说明该阶段内部使用了哪些观测变体和地图变体；同时输出 milestone 和 progress，用于展示物品搜集、开箱、击杀、进入新房间等阶段性进展。

## 六、JSON 输出结构

使用 `--json-out` 后，输出文件包含两个顶层字段：

```json
{
  "summary": {},
  "episodes": []
}
```

`summary` 是按任务和测评阶段聚合后的统计结果；`episodes` 是每个 episode 的详细结果。每个 episode 包含：

| 字段 | 含义 |
|---|---|
| `task_id` | 任务 ID |
| `eval_stage` | 测评阶段 |
| `obs_variant` | 观测变体 |
| `map_variant` | 地图变体 |
| `seed` | 随机种子 |
| `steps` | 执行步数 |
| `total_reward` | 累计奖励 |
| `terminated` | 是否自然终止 |
| `truncated` | 是否因步数上限截断 |
| `success` | 是否完成任务 |
| `terminal_reason` | 终止原因 |
| `event_counts` | episode 内事件计数 |
| `milestones` | 关键里程碑是否达成 |

`summary` 中的 `progress_rates` 表示某类进展事件在该阶段 episode 中出现过的比例。例如：

```json
{
  "key_collected": 0.8,
  "chest_opened": 0.6,
  "monster_killed": 0.4
}
```

表示 80% 的 episode 收集过钥匙，60% 打开过宝箱，40% 击杀过怪物。它们反映的是阶段性进展，不等同于最终成功率。

## 七、测试内容：观测变体

`--obs-variants` 控制测评脚本在把像素观测传给 policy 前进行何种变换。该变换只发生在测评脚本内部：

- 不改变 `nesylink` 环境本体。
- 不改变地图、奖励、碰撞和任务终止条件。
- 不改变 `env.render()` 的真实渲染。
- 不影响训练脚本或其他 agent

基础用法：

```bash
python utils/evaluate_policy.py \
  --policy submissions/student_policy.py \
  --obs-variants default grayscale dark bright high_contrast inverted
```

当前支持的颜色/亮度变体：

| 变体 | 说明 |
|---|---|
| `default` | 原始像素观测，不做修改 |
| `grayscale` | 转为灰度图后复制为 3 通道 RGB |
| `dark` | 整体变暗 |
| `bright` | 整体变亮 |
| `high_contrast` | 按阈值二值化，形成高对比图 |
| `inverted` | RGB 反色 |

## 八、测试内容：重画渲染方法

除颜色变换外，测评脚本还支持“重画”类变体。它们不会从原始像素图中识别对象，而是读取当前环境状态，将地图重新绘制成另一套视觉符号后再传给 policy

使用方式：

```bash
python utils/evaluate_policy.py \
  --policy submissions/student_policy.py \
  --obs-variants default redraw_geometric redraw_symbols
```

### 方案一

使用几何形状替代原始 sprite：

| 对象 | 新图像表示 |
|---|---|
| floor | 深色网格背景 |
| wall | 白色实心方块 |
| chest | 黄色菱形 |
| agent | 青色圆形，内部三角表示朝向 |
| monster | 红色六边形 |
| trap | 紫色 X |
| button | 绿色圆点 |
| switch | 橙色菱形 |
| exit | 蓝色门框 |
| gap | 黑色圆洞 |
| bridge | 棕色桥板 |
| npc | 粉色十字 |

预览：

![redraw_geometric](assets/redraw_geometric.png)

### 方案二


| 对象 | 新图像表示 |
|---|---|
| floor | 浅灰背景 |
| wall | 黑色实心方块 |
| chest | 金色方框加中央标记 |
| agent | 白色三角箭头 |
| monster | 红色符号块 |
| trap | 紫色警告三角 |
| button | 绿色圆点 |
| switch | 橙色斜杠 |
| exit | 蓝色门框 |
| gap | 黑色空洞 |
| bridge | 棕色桥板 |
| npc | 粉色简化人形 |

预览：

![redraw_symbols](assets/redraw_symbols.png)

## 九、固定比例鲁棒性套件

固定比例鲁棒性套件支持两种 policy 提交方式。

方式一：所有 task 使用同一个统一 agent。适合实现了多任务策略、根据 `task_id` 或视觉输入自动切换行为的提交：

```bash
python utils/evaluate_policy.py \
  --policy submissions/student_policy.py \
  --robustness-suite \
  --num-envs 100 \
  --json-out results/robustness_suite_eval.json
```

方式二：每个 task 指定一个单独的 agent 文件。适合每关分别设计策略、分别训练模型或分别保存权重的提交：

```bash
python utils/evaluate_policy.py \
  --tasks \
    mathematical_logic/task_1 \
    mathematical_logic/task_2 \
    mathematical_logic/task_3 \
    mathematical_logic/task_4 \
    mathematical_logic/task_5 \
  --task-policy mathematical_logic/task_1=submissions/task1_agent.py \
  --task-policy mathematical_logic/task_2=submissions/task2_agent.py \
  --task-policy mathematical_logic/task_3=submissions/task3_agent.py \
  --task-policy mathematical_logic/task_4=submissions/task4_agent.py \
  --task-policy mathematical_logic/task_5=submissions/task5_agent.py \
  --robustness-suite \
  --num-envs 100 \
  --json-out results/robustness_suite_eval.json
```

也可以混合使用：用 `--policy` 指定默认统一 agent，再用 `--task-policy` 覆盖其中某几个 task：

```bash
python utils/evaluate_policy.py \
  --policy submissions/shared_agent.py \
  --task-policy mathematical_logic/task_4=submissions/task4_specialist.py \
  --task-policy mathematical_logic/task_5=submissions/task5_specialist.py \
  --robustness-suite \
  --num-envs 100
```

启用 `--robustness-suite` 后，`--num-envs` 表示每个 task 的总 episode 数。脚本会按任务切分为不同测评阶段：

| 任务 | `original` | `spatial` | `color` | `redraw` |
|---|---:|---:|---:|---:|
| Task 1-3 | 50% | 30% | 10% | 10% |
| Task 4-5 | 80% | 0% | 10% | 10% |

例如 `--num-envs 100` 时：

| 任务 | `original` | `spatial` | `color` | `redraw` |
|---|---:|---:|---:|---:|
| Task 1-3 | 50 | 30 | 10 | 10 |
| Task 4-5 | 80 | 0 | 10 | 10 |

正式实验报告建议使用默认的 `--num-envs 100` 或更高轮数；如果只是检查脚本和 policy 接口是否能跑通，可以临时传更小的值做 smoke test。

阶段含义：

| 阶段 | 使用的观测变体 | 使用的地图变体 | 说明 |
|---|---|---|---|
| `original` | `default` | `default` | 原始地图和原始渲染输入 |
| `spatial` | `default` | `spatial_a/b/c` 循环使用 | 仅 Task 1-3 使用，扰动 spawn、障碍、怪物或宝箱位置 |
| `color` | `grayscale/dark/bright/high_contrast/inverted` 循环使用 | `default` | 改变颜色、亮度和对比度 |
| `redraw` | `redraw_geometric/redraw_symbols` 交替使用 | `default` | 改变角色、物体和地图的绘制方式 |

输出 summary 会按阶段给出成功率和进展：

```text
mathematical_logic/task_3 [original]
  episodes:     10
  success_rate: 0.900
  variants:     {'default': 10}
  map_variants: {'default': 10}
  milestones:
    monster_killed: 1.000
    key_collected: 0.900
  progress:
    monster_killed: 1.000
    key_collected: 0.900
```

最终实验报告应按 task 和阶段分别报告成功率：

| 任务 | 报告中必须包含的阶段成功率 |
|---|---|
| Task 1 | `original`、`spatial`、`color`、`redraw` |
| Task 2 | `original`、`spatial`、`color`、`redraw` |
| Task 3 | `original`、`spatial`、`color`、`redraw` |
| Task 4 | `original`、`color`、`redraw` |
| Task 5 | `original`、`color`、`redraw` |

其中 Task 1-3 的 `spatial` 阶段用于评估坐标扰动后的泛化能力；Task 4-5 不运行 `spatial` 阶段，因为任务难度更高，鲁棒性测试只覆盖颜色变化和绘制方式变化。




## 十一、注意事项

- `obs` 的 shape 保持为 `(128, 160, 3)`，dtype 保持为 `uint8`。
- 固定比例套件按 `(task_id, stage)` 单独统计 summary。
- Task 1-3 的 `spatial` 地图是在测评时临时生成的，不会修改仓库中的原始地图 JSON。
