# CWVL-USV 公开复现流程

本文档给出从障碍物地图、KF replay cache、五训练种子训练到统一评估的执行顺序。所有命令均从仓库根目录运行。

## 1. 障碍物地图

正式地图覆盖 obs3–10，并包含 PPO-family 训练使用的 obs3–6 混合评估集。地图文件可由下列生成器重建。

obs3 的训练集和评估集使用固定生成参数：

```powershell
python simple_boat/map_generator/Generate_training_new.py --n-obs 3 --count 500 --output-dir simple_boat/assets/nav3_new_map --prefix nav3_new_b00 --seed 20260603
python simple_boat/map_generator/Generate_training_new.py --n-obs 3 --count 200 --output-dir simple_boat/assets/eval3_new_map --prefix eval3_new_b00 --seed 20261603
```

obs4--6 使用统一包装器生成或核验训练集和评估集：

```powershell
python scripts/generate_final_maps.py
```

obs7–10 使用统一入口按顺序生成或核验评估地图：

```powershell
python scripts/generate_obs7_10_maps.py
```

## 2. KF replay cache

使用 `scripts/precompute_kf_cache.py` 为每个场景目录预计算 cache。obs3–6 使用 `configs/evaluation/delay20_cov100_binned_s10.yaml`；obs7–10 的场景目录、评估配置和 cache 目录由实验协议文件统一给出。

单个目录的标准命令为：

```powershell
python scripts/precompute_kf_cache.py --config <evaluation-config> --scenario-dir <scenario-dir> --cache-dir <cache-dir> --cache-mode read_write --all-binned-candidates --forced-seed 0 --workers 4 --audit-output <audit.json>
```

obs3–6 依次处理各训练和评估目录；obs7–10 依次处理各自的评估目录，并写入实验协议声明的 `kf_cache_dir`。训练或评估前，再以 `--cache-mode read_strict` 运行同一命令进行完整性核验。

PPO-family 的四个训练源和 mixed eval bundle 可用正式配置统一检查；脚本会直接读取每个 `train_sources` 的地图与 cache 映射，不依赖额外的 mixed training 地图副本：

```powershell
python scripts/check_mixed_cache_hits.py --workers 8 --chunk-size 50
```

## 3. 五训练种子

PPO、Hetero-PPO 与 CWVL 共 15 个训练任务：

```powershell
python scripts/run_ppo_family_multiseed.py --seeds 0 1 2 3 4 --max-parallel-training 1
```

DRL-VO 与 Lag-U 共 10 个训练任务：

```powershell
python scripts/run_drl_vo_lag_u_multiseed.py --seeds 0 1 2 3 4 --max-parallel-training 2
```

两个入口都支持 `--dry-run`，可在长时间训练前检查完整命令矩阵。每个底层训练器保存 resolved config、manifest、阶段模型和恢复状态；重复运行时会跳过完整任务，DRL-VO/Lag-U 会从现有 trainer state 继续未完成训练。

## 4. 统一评估

PPO family 使用同一个评估入口；以下命令对一个已经确定的模型执行 200 回合 cache 评估：

```powershell
python scripts/eval_transfer.py --config configs/evaluation/delay20_cov100_binned_s10.yaml --checkpoint <model.zip> --vec-normalize <vec_normalize.pkl> --scenario-dir <scenario-dir> --cache-dir <cache-dir> --cache-mode read_strict --episodes 200 --seed 0 --deterministic --label <method_obs> --output-dir <output-dir>
```

DRL-VO 与 Lag-U 使用同一份 obs3--10 协议：

```powershell
python scripts/eval_dqn_vo_baseline.py --config configs/experiments/dqn_vo_baseline.yaml --model <model.zip> --vecnormalize <vecnormalize.pkl> --protocol configs/experiments/drl_vo_lag_u_multiseed_obs3_10.json --episodes 200 --seed 0 --output-dir <output-dir>
python scripts/eval_lag_u_baseline.py --config configs/experiments/lag_u_baseline.yaml --model <model.pt> --protocol configs/experiments/drl_vo_lag_u_multiseed_obs3_10.json --episodes 200 --seed 0 --output-dir <output-dir>
```

所有方法使用相同场景、评估 seed、回合数和指标定义。成功率以全部回合为分母；速度与路径长度只在成功回合上统计；在线延迟按实际控制步数统计。

## 5. COLREGs-MPCC

COLREGs-MPCC 是独立的确定性控制器，配置和实现分别位于 `configs/experiments/colregs_mpc_baseline.yaml` 与 `scripts/eval_colregs_mpc_baseline.py`：

```powershell
python scripts/eval_colregs_mpc_baseline.py --config configs/experiments/colregs_mpc_baseline.yaml --episodes 200 --seed 0 --workers 1 --output-dir <output-dir>
```

运行完成后应保留每个密度的 episode CSV、summary CSV、resolved config 和 manifest。公开结果表位于本目录的 `six_algorithm_comparison.md`、`proposed_method_comparison.md` 与 `显著性实验.md`。
