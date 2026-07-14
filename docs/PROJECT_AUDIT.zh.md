# 项目梳理与 Git 状态

## 1. 当前仓库状态

- 当前分支：`feature/pixel2state`
- 工作区状态：干净，无未提交改动
- 主要本地分支：
  - `main`
  - `feature/pixel2state`
- 远端分支：
  - `origin/main`
  - `origin/feature/pixel2state`
  - `origin/leanBranch`

## 2. 分支关系概览

- `main` 是环境主线，核心内容是 `nesylink` 游戏环境、任务、奖励和文档。
- `feature/pixel2state` 在主线基础上增加了像素到状态的训练、推理和提交策略代码。
- `leanBranch` 更早，当前看起来不是主要开发分支。

`feature/pixel2state` 相比 `main` 的主要新增：

- `experiments/pixel2state/`
  - 数据集采集
  - 卷积模型定义
  - 训练脚本
  - 推理脚本
  - 评估脚本
- `submissions/`
  - `task3_agent.py`
  - `task4_agent.py`
  - `common/` 共享推理与工具函数
- `data/pixel2state/train.npz`
- `models/pixel2state/best.pt`
- `models/pixel2state/eval_results.json`
- `utils/human_play.py`
- `utils/evaluate_policy.py`

## 3. 项目结构

### 3.1 环境主干

- `nesylink/env.py`
  - 项目总入口
  - 负责把任务配置、地图、奖励、wrapper 拼装成环境
- `nesylink/__init__.py`
  - 导出 `make_env`
  - 导入时自动注册 Gym 环境
- `nesylink/game.py`
  - 基于 pygame 的人工调试入口

### 3.2 任务与奖励

- `nesylink/tasks/`
  - `specs.py`：任务数据结构 `TaskSpec`
  - `registry.py`：任务注册与查询
  - `builtin.py`：内置 task1-task5
- `nesylink/rewards/`
  - `loader.py`：根据 `reward_id` 或模块路径装载奖励
  - `mathematical_logic/`：五个任务的奖励逻辑

### 3.3 核心引擎

- `nesylink/core/mechanics/engine.py`
  - 游戏状态推进核心
  - 处理移动、交互、战斗、死亡、通关
- `nesylink/core/observation.py`
  - 负责把运行时状态编码成网格或结构化观测
- `nesylink/core/info.py`
  - 负责构造 `info`
  - 暴露 episode、inventory、entities、events、debug 等信息
- `nesylink/core/world/`
  - 地图 JSON 加载、解析、校验
- `nesylink/core/rendering/`
  - 像素画面渲染

### 3.4 wrapper 与对外接口

- `nesylink/wrappers/gym_env.py`
  - Gymnasium 环境实现
  - 支持 `full` / `grid` / `pixels` 三类观测
  - 支持 `pixel` / `grid` 两种控制模式

## 4. 运行链路

项目的主要调用链大致是：

`make_env(...)`
-> 解析 `task_id` / `map_id` / `reward_id`
-> 加载地图与奖励
-> 创建 `GymDungeonEnv`
-> 内部创建 `DungeonEngine`
-> `reset()` / `step()` 推进运行时状态
-> 生成 `obs` 与 `info`

这条链路设计得比较清晰，职责分层也比较明确：

- 地图 JSON 管世界内容
- `tasks` 管任务装配
- `rewards` 管奖励与任务终止补充逻辑
- `core` 管真正的游戏规则
- `wrappers` 管 RL 接口

## 5. `feature/pixel2state` 分支重点

### 5.1 目标

这个分支的目标很明确：在只能看像素输入的条件下，先把观测还原成语义网格，再基于恢复出的状态做策略。

### 5.2 主要实现

- `experiments/pixel2state/dataset.py`
  - 通过同时跑 `pixels` 环境和 `full` 环境，采集像素和网格标签对
- `experiments/pixel2state/model.py`
  - 轻量 CNN，把 `128x160x3` 映射到 `8x10` 语义网格
- `experiments/pixel2state/train.py`
  - 训练入口
  - 包含亮度、颜色、灰度、反色、量化、噪声等增强
- `experiments/pixel2state/infer.py`
  - 评估期像素扰动与 redraw 适配的核心推理逻辑
- `submissions/task3_agent.py`
  - task3 策略
- `submissions/task4_agent.py`
  - task4 策略

### 5.3 这一分支的设计特点

- 优点：
  - 训练链路完整，从数据采集到模型到策略闭环齐全
  - 明显在针对鲁棒性评测做适配
  - `utils/evaluate_policy.py` 已内置颜色扰动、重绘和空间扰动评估
- 风险：
  - `submissions/` 下有一些代码直接依赖 `experiments/`，提交代码和实验代码耦合较紧
  - 部分中文注释存在编码问题，后续协作阅读成本会偏高
  - agent 里有较多启发式硬编码，后期维护时要小心任务地图变体

## 6. 当前 Git 判断

从提交历史看：

- `main` 更像稳定主线
- `feature/pixel2state` 是功能开发分支，提交集中在鲁棒性、评估和 task3/task4 agent
- 当前工作区干净，适合继续做下面几类 Git 操作：
  - 继续在 `feature/pixel2state` 上迭代
  - 整理后合并回 `main`
  - 或拆出更细的子分支继续重构

## 7. 建议的 Git 工作流

### 7.1 如果你要继续做实验与提交策略

建议保留：

- `main`：稳定环境主线
- `feature/pixel2state`：像素识别与提交策略主分支

继续新增功能时，建议从 `feature/pixel2state` 再切子分支，例如：

- `feature/pixel2state-refactor`
- `feature/task4-policy`
- `feature/eval-robustness`

### 7.2 如果你要准备合并回主线

建议先做这几件事：

1. 清理编码异常的注释和文档
2. 把 `submissions` 对 `experiments` 的强依赖降下来
3. 检查大文件是否适合继续直接进 Git
   - `data/pixel2state/train.npz`
   - `models/pixel2state/best.pt`
4. 补一轮最基本的可复现实验说明

### 7.3 大文件管理建议

目前分支里已经直接纳入了数据和模型文件。后续如果这些文件会持续变大，建议考虑：

- 使用 Git LFS
- 或把数据/模型改为产物，不直接长期版本化

## 8. 后续最值得做的整理

优先级建议如下：

1. 把 `PixelToStatePredictor` 抽成稳定公共模块，避免 `submissions` 直接引用实验脚本
2. 给 `submissions/task3_agent.py` 和 `submissions/task4_agent.py` 做一次职责拆分
3. 为 `feature/pixel2state` 补一份专门 README，写清训练、评估、提交方式
4. 视情况把模型与数据迁移到 LFS 或外部存储
