# Go2-X5 Navigation Pipeline

这个 README 只保留当前导航主链路的操作说明，目标是从一份 `usda` 场景入口文件开始，串起下面这条流程：

`USDA -> collision/visual 资产检查 -> 2D occupancy grid -> 起终点点选 -> A* 全局路径 -> DWA 局部控制 -> RL locomotion 执行 -> 轨迹日志`

## 0. 前提

如果你手里只有一份场景入口文件，例如：

- `/home/y/usda/839955.usda`

先确认它引用的外部资产还在。当前这类 `usda` 通常不是自包含文件，而是一个“装配入口”。

以 `/home/y/usda/839955.usda` 为例：

- 视觉层 `gauss` 来自 `/home/y/usdz/839955.usdz`
- 碰撞层 `scene_collision` 来自 `/home/y/Collision_Mesh/839955/839955_collision.usd`

如果这两个引用文件不存在：

- `play_cs.py` / `play_nav_cs.py` 里的场景显示会异常
- `scene_collision` 无法正常提供地面和碰撞几何
- 从 `usda` 重新导出导航地图也会失败或不完整

所以从零开始之前，先确认：

- `/home/y/usda/839955.usda`
- `/home/y/usdz/839955.usdz`
- `/home/y/Collision_Mesh/839955/839955_collision.usd`

都存在，或者你已经把 `usda` 里的引用路径改对。

## 1. 先验证基线场景

第一步不要急着做导航，先确认场景和机器人在 Isaac 里能正常加载。

```bash
python scripts/reinforcement_learning/rsl_rl/play_cs.py \
  --task RobotLab-Isaac-Velocity-Flat-Go2-X5-ArmUnlock-v0 \
  --checkpoint /home/y/DWA/flat/model_8500.pt \
  --num_envs 1 \
  --map /home/y/usda/839955.usda
```

通过标准：

- 场景能正常显示
- 机器狗正常落地，不悬空
- 机器狗前进命令有效

同一条命令也可以用来可视化机器狗导航与小物体交互的功能：

```bash
python scripts/reinforcement_learning/rsl_rl/play_cs.py \
  --task=RobotLab-Isaac-Velocity-Flat-Go2-X5-ArmUnlock-v0 \
  --checkpoint=flat/model_8500.pt \
  --num_envs=1 \
  --map=/home/y/usda/839955.usda
```

如果这一步不通过，不要继续做地图和导航，先修场景资产依赖。

## 2. 从 `scene_collision` 导出 2D 导航地图

导航地图来自碰撞层，不是视觉层。

```bash
python scripts/navigation/export_nav_map.py \
  /home/y/usda/839955.usda \
  --prim-path /World/scene_collision \
  --resolution 0.05 \
  --min-obstacle-height 0.15 \
  --max-obstacle-height 2.5
```

默认输出目录类似：

```text
/home/y/usda/nav_maps/839955/
```

会生成：

- `occupancy.pgm`
- `map.json`
- `preview.ppm`

这些文件的含义：

- `occupancy.pgm`：给算法使用的二值占据栅格
- `map.json`：分辨率、原点、图像路径等元数据
- `preview.ppm`：给人检查用的预览图

## 3. 检查地图和坐标系

先不要直接猜起点终点坐标。先用辅助脚本确认地图范围和空地点。

```bash
python scripts/navigation/inspect_nav_map.py \
  --map /home/y/usda/nav_maps/839955/map.json \
  --clearance 0.15 \
  --spacing 0.6 \
  --list-free 20
```

这个脚本会输出：

- 世界坐标范围
- 地图原点
- 一批带安全边距的候选空地点

如果你想查询某个具体 world 点是否在空地：

```bash
python scripts/navigation/inspect_nav_map.py \
  --map /home/y/usda/nav_maps/839955/map.json \
  --world -2.03488 5.00164 \
  --list-free 0
```

如果你想直观看不同 `inflate-radius` 会把障碍“吃胖”到什么程度，可以先生成膨胀对比图：

```bash
python scripts/navigation/visualize_inflation.py \
  --map /home/y/usda/nav_maps/839955/map.json \
  --radii 0.15 0.2 0.25 0.3
```

默认会在 `map.json` 同目录下生成：

- `inflation_preview/raw_map.ppm`
- `inflation_preview/inflate_*.ppm`
- `inflation_preview/comparison.ppm`
- `inflation_preview/manifest.json`

如果你想单独指定输出目录：

```bash
python scripts/navigation/visualize_inflation.py \
  --map /home/y/usda/nav_maps/839955/map.json \
  --radii 0.18 0.30 0.32 0.40 \
  --output-dir /tmp/839955_inflation_preview
```

如果你想继续看“给定 `inflate-radius` 后，A* 实际规划出来的路径是什么样，以及留给 DWA 的跟踪点有哪些”，可以运行：

```bash
python scripts/navigation/visualize_astar_dwa.py \
  --map /home/y/usda/nav_maps/839955/map.json \
  --start 3.335 5.435 \
  --goal -2.915 6.185 \
  --inflate-radius 0.30
```

如果你想指定输出目录：

```bash
python scripts/navigation/visualize_astar_dwa.py \
  --map /home/y/usda/nav_maps/839955/map.json \
  --start 3.335 5.435 \
  --goal -2.915 6.185 \
  --inflate-radius 0.30 \
  --output-dir /tmp/839955_astar_dwa_vis
```

这个脚本会输出：

- `astar_preview.ppm`：膨胀后的地图 + A* 原始栅格路径 + prune 后折线路径
- `dwa_points_preview.ppm`：在 A* 路径基础上，再叠加实际提供给 DWA 的加密跟踪点
- `summary.json`：A* 原始点数、pruned waypoint 数、DWA 跟踪点数等统计

颜色约定：

- 深色：膨胀后的障碍
- 红色：A* 原始栅格路径
- 橙色：A* prune 后保留下来的折线 waypoint
- 青色：DWA 实际跟踪的加密路径点
- 绿色：起点
- 蓝色：终点

注意：

- 这个脚本里的 `--start` 使用的是你手动传入的世界坐标
- `play_nav_cs.py` 真正在线规划时，起点是 `settle` 之后机器人的实际位置
- 所以如果你要和在线运行完全一一对应，最好使用 `settle` 后的真实起点来可视化

## 4. 用 `pick_nav_points.py` 在预览图上点选起终点

如果你不想手输坐标，推荐直接用点选脚本。

脚本路径：

- [pick_nav_points.py](/home/y/DWA/scripts/navigation/pick_nav_points.py)

推荐命令：

```bash
python scripts/navigation/pick_nav_points.py \
  --map /home/y/usda/nav_maps/839955/map.json \
  --clearance 0.32 \
  --scale 4 \
  --output /home/y/usda/nav_maps/839955/picked_points_1.json
```

如果你想点选某张膨胀预览图，而不是默认的 `preview.ppm`，可以显式传入：

```bash
python scripts/navigation/pick_nav_points.py \
  --map /home/y/usda/nav_maps/839955/map.json \
  --preview /home/y/usda/nav_maps/839955/inflation_preview/inflate_0p32.ppm \
  --clearance 0.32 \
  --scale 4 \
  --output /home/y/usda/nav_maps/839955/picked_points_0p32.json
```

使用时：

- 左键添加点
- 右键撤销最后一个点
- `C` 清空当前点
- `S` 保存到 `--output`
- `Q` / `Esc` 退出

保存的 JSON 会记录：

- 栅格坐标 `grid.row / grid.col`
- 世界坐标 `world.x / world.y`
- 原始地图下是否空闲 `raw_free`
- 按 `--clearance` 膨胀后是否仍然安全 `clearance_free`

## 5. 在仿真里运行 `play_nav_cs.py`

```bash
python scripts/reinforcement_learning/rsl_rl/play_nav_cs.py \
  --task RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0 \
  --checkpoint /home/y/DWA/flat/model_8500.pt \
  --num_envs 1 \
  --map /home/y/usda/839955.usda \
  --nav-map /home/y/usda/nav_maps/839955/map.json \
  --goal -2.915 6.185 \
  --goal-yaw 1.57 \
  --goal-yaw-tolerance 0.01 \
  --spawn 3.335 5.435 3.14 \
  --inflate-radius 0.3 \
  --local-clearance-radius 0.25 \
  --settle-steps 120 \
  --debug-print-every 20 \
  --max-steps 3000 \
  --head-camera \
  --goal-tolerance 0.35 \
  --dataset-dir /home/y/DWA/episodes
```

当前脚本会做这些事：

1. 用 `scene_collision` 建场景和碰撞环境
2. 让机器人先 `settle`
3. 从落稳后的真实位姿做一次全局 A* 规划
4. 用 DWA 生成局部 `vx, 0, wz` 命令并交给 RL locomotion policy 执行
5. 按 episode 格式保存数据（见第 7 节）

## 6. 看什么日志算正常

`play_nav_cs.py` 运行时会打印 `[NAV-CS]` 日志，主要字段含义：

- `pose=(x, y, yaw)`：机器人当前位姿
- `z=`：机器人 base 高度
- `cmd=(vx, vy, wz)`：DWA 输出给 `base_velocity` 的命令
- `measured=(vx, wz)`：机器人真实运动速度
- `goal_dist=`：当前到目标点的距离
- `clearance=`：当前 DWA 选中轨迹的局部安全距离估计
- `settling=`：是否还在稳定阶段
- `debug_cmd=`：是否在固定命令调试模式

一轮正常导航通常表现为：

- `settling=True` 时 `cmd` 为 `0`
- 进入导航后 `goal_dist` 整体下降
- `measured` 与 `cmd` 大方向一致
- `clearance` 保持正数

## 7. Episode 数据格式

每次运行 `play_nav_cs.py` 会在 `--dataset-dir`（默认 `<repo>/episodes`）下生成一个 episode，目录结构如下：

```text
episodes/
└── 1/                      ← 任务编号（固定，由 --task-id 指定，默认 1）
    ├── 1/                  ← 第 1 条轨迹（自动递增）
    │   ├── 1-1/            ← 子任务 1：从出发点导航到目标位置
    │   │   ├── data.csv
    │   │   └── images/front/
    │   │       ├── camera0_00000.jpg
    │   │       └── ...
    │   └── 1-2/            ← 子任务 2：在目标位置调整 yaw（仅当传入 --goal-yaw 时生成）
    │       ├── data.csv
    │       └── images/front/
    │           ├── camera0_00000.jpg
    │           └── ...
    ├── 2/                  ← 第 2 条轨迹
    │   ├── 2-1/
    │   └── 2-2/
    └── 3/                  ← 第 3 条轨迹（依此类推）
        ...
```

**子任务划分规则：**

| 子任务 | 触发条件 |
|--------|----------|
| `N-1` 导航阶段 | settle 结束后 → 到达目标位置（`distance_to_goal ≤ goal-tolerance`） |
| `N-2` 对齐阶段 | 到达目标位置后 → 完成偏航角对齐（仅在指定 `--goal-yaw` 时存在） |

**`data.csv` 列定义：**

```
时间戳(秒),位置X,位置Y,位置Z,姿态X,姿态Y,姿态Z,姿态W,
线速度X,线速度Y,线速度Z,
关节1,关节2,关节3,关节4,关节5,关节6,夹爪,前摄像头图像
```

| 字段 | 含义 |
|------|------|
| `时间戳(秒)` | `step × dt`，仿真时间，50 Hz |
| `位置X/Y/Z` | base link 世界坐标（m） |
| `姿态X/Y/Z/W` | 四元数（x, y, z, w），世界系 |
| `线速度X/Y/Z` | base link body frame 线速度（m/s） |
| `关节1-6` | `arm_joint1` ~ `arm_joint6`（rad） |
| `夹爪` | `arm_joint7` 与 `arm_joint8` 的平均值（rad） |
| `前摄像头图像` | 对应帧图像文件名（相对于 `images/front/`） |

**图像：**

- 格式：JPEG，quality=90
- 分辨率：由 `--head-camera-height` / `--head-camera-width` 控制（默认 480×640）
- 频率：与仿真控制频率相同，50 Hz（每 env.step 一帧）
- 命名：每子任务独立从 `camera0_00000.jpg` 开始计数

**仿真频率参考：**

| 层级 | 频率 |
|------|------|
| 物理步 `sim.dt` | 200 Hz（0.005 s） |
| 控制步 `env.step`（decimation=4） | 50 Hz（0.02 s） |
| 数据采集（图像 + CSV 行） | 50 Hz |

**指定数据集目录：**

```bash
--dataset-dir /home/y/DWA/episodes
```

- `--task-id` 固定本次采集属于哪个任务，默认为 `1`，不同任务类型可指定不同编号
- `--dataset-dir` 不传时默认为 `<repo>/episodes/`
- 每次运行会扫描 `<dataset-dir>/<task-id>/` 下已有的数字子目录，取 `max + 1` 作为新轨迹编号，保证不覆盖历史数据

## 8. 当前系统状态

当前已经打通的链路：

- 从 `usda` 导出导航地图
- 基于 occupancy grid 做 A*
- 用 `pick_nav_points.py` 点选起终点
- 在仿真里使用 `play_nav_cs.py` 进行 A* + DWA + RL locomotion 在线导航
- 机器狗头顶 RGB 摄像头（`base` link 前方 28 cm、上方 7 cm，朝向正前方）
- 按子任务结构（导航阶段 / yaw 对齐阶段）保存 episode：CSV 状态 + JPEG 图像，50 Hz

当前仍在继续优化的部分：

- DWA 的终点附近收敛
- 任务成功判定阈值
- 更稳定的终端接近策略
- 批量任务调度与自动采集


## 9. 推荐验证顺序

从头串流程时，严格按下面顺序做：

1. 先检查 `usda` 依赖资产是否存在
2. 先跑 `play_cs.py`，确认场景和机器人正常
3. 再导出 `nav map`
4. 再用 `inspect_nav_map.py` / `pick_nav_points.py` 选点
5. 再做离线 A*
6. 最后才跑 `play_nav_cs.py`

不要跳步骤。当前最容易浪费时间的情况，是场景资产没对齐时直接去调导航控制器。
