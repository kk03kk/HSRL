# SERVER → LOCAL 交接：实机切到 final 14858（2026-11-11）

```yaml
status: pending-local
owner: server
updated: 2026-11-11 23:20 +0800
base_commit: b6c4624（推送后以实际 commit 为准）
```

## 1. 模型选择

- **冠军/实机模型：`league-final-14858`**（E1 点估计第一，用户确认直接使用）。
- E1（2,688 局，valid/eligible）：final **1086.53**（95% CI [1062.49, 1111.01]；
  排名 CI [1,2]），13806 1068.45，12311 1055.92，V1.1 1000。
- final 对 13806 为 240–208（53.57%，CI 跨 0.5），不是显著超越；按用户规则
  **E1 只跑一轮、不做冠军挑战赛**，点估计第一即冠军。
- 模型：`artifacts/models/league-final-20261111-step014858/`
  - `model.pt` SHA256 `ba7e8bbcf69f3f6142041369de3f9930242bae2de5c826c205c6d509b5dbd8e7`
  - 参数 16,436,034；arch `entity`；训练引擎 `c4c71014e3d41760`；
    酒馆 `37.0.2.256589`；动作 245；观测 3227 + 124 cards。
- 不把 `trainer.pt` 给实机；live 包只含 model/manifest/checkpoint/MODEL_CARD。

## 2. 服务器侧 live 包

```text
~/projects/HSAI/artifacts/exports/league-final-14858-live.tgz
size  ~54 MB
SHA256 32ccdabdc7d43793fb7d358185741fbf082ba0f5c40a397d7967ade01859a22a
```

包内：`model.pt`、`manifest.json`、`checkpoint.json`、`MODEL_CARD.md`。

## 3. 本地下载命令（Windows PowerShell）

把 `<nus-login>` 换成你现有的 NUS SSH 别名/主机（例如你平时 `ssh <nus-login>` 用的那个）：

```powershell
# 1) 下载包和校验文件
scp <nus-login>:~/projects/HSAI/artifacts/exports/league-final-14858-live.tgz `
    D:\HSAI\artifacts\exports\
scp <nus-login>:~/projects/HSAI/artifacts/exports/league-final-14858-live.tgz.sha256 `
    D:\HSAI\artifacts\exports\

# 2) 校验（必须是 32ccdabd…）
Get-FileHash D:\HSAI\artifacts\exports\league-final-14858-live.tgz -Algorithm SHA256
Get-Content D:\HSAI\artifacts\exports\league-final-14858-live.tgz.sha256
```

## 4. 本地解包与 pin（本地写权）

```powershell
$dst = 'D:\HSAI\artifacts\models\league-final-20261111-step014858'
New-Item -ItemType Directory -Force -Path $dst | Out-Null
tar -xzf D:\HSAI\artifacts\exports\league-final-14858-live.tgz -C $dst
Get-FileHash "$dst\model.pt" -Algorithm SHA256
# 必须是 ba7e8bbcf69f3f6142041369de3f9930242bae2de5c826c205c6d509b5dbd8e7
```

更新 `config/live/model.json`（**本地文件，服务器不改**）。建议内容：

```json
{
  "schema": "hsai.live-model/v1",
  "model_id": "hsai-final-14858",
  "checkpoint": "artifacts/models/league-final-20261111-step014858/model.pt",
  "sha256": "ba7e8bbcf69f3f6142041369de3f9930242bae2de5c826c205c6d509b5dbd8e7",
  "trained_game_version": "37.0.2.256589",
  "trained_engine_source_hash": "c4c71014e3d41760",
  "required_client_build": 255790,
  "client_build_basis": "本机已安装客户端：Hearthstone.exe sha256 与 profile.executable_sha256 一致，Power.log 记录 BuildNumber=255790；37.0.2.256589 是训练所用模拟器规则版本，不是实时客户端构建号",
  "selection_reason": "E1 四模型校准 2,688 局点估计第一（1086.53，CI [1062.49,1111.01]，排名 CI [1,2]）；用户确认直接作为实机冠军模型；不做挑战赛",
  "parameters": 16436034,
  "engine_cards_sha256": "7a02d10794cf80c56fa77b1dd235d49b892c8d3e31c2cf56887f696e2cf922a6",
  "real_game_accepted": false
}
```

- `config/live/engine.json` **不需要改**：仍是 `c4c71014e3d41760`、Windows wheel
  `524d6871…`；本次没有引擎变化。
- `real_game_accepted` 在 E-0/E-1 通过前保持 `false`。

## 5. 本地验收顺序

1. **doctor + 推理自检**
   ```powershell
   python client/live_245.py doctor --game-dir D:\Hearthstone --infer
   ```
   确认模型 SHA、引擎 pin、动作/观测维度、`hsbg_engine` 构建一致；不一致只告警，
   `require_match` 仍按现有配置。

2. **E-0 canary：1 局**
   ```powershell
   python client/live_245.py autoplay --game-dir D:\Hearthstone `
     --out artifacts/runs/live-final14858-e0-<date>-001 --seconds 3600 --games 1
   ```
   产物至少保留：`decisions.jsonl`、`decisions-full.jsonl.gz`、`actions.jsonl`、
   `summary.json`、`performance.json`、`health.json`、`result.json`、结果屏截图、
   `session-archives/`。

3. E-0 通过标准：S0=0、无非法/未授权动作、无 repeated blocked、名次/结果屏一致、
   动作速率正常、无 supervisor 异常。通过后再按现有 E 阶段手册做 E-1 三局。

## 6. 实机风险提醒（D-026）

- final 是在**完整模拟器观测**上训练的；本地 live 投影仍有已知近似：
  对手已见棋盘/最后对手、quests/owned_rewards、ChooseOne/Target/Timewarp 分类、
  多个 `MODEL_*` 全局字段、Freeze toggle、>16 选择、trinket/Chronum 可用性。
  详见 `docs/sync/DECISIONS.md` D-026。
- 因此 E-0 首先看**工程可靠性**（不 blocked、不非法、不卡死），不要把 E-0 的排名
  当作棋力结论；棋力仍以服务器 E1 为准。
- final 的 Freeze 倾向较高（服务器上约 22% 决策；`forbid_unfreeze` 规则可降到 ~8%）。
  本地桥接在全冻结时不暴露 Freeze，实际会自然抑制一部分；若 E-0 出现决策速率异常
  或大量 Freeze，保留日志回传。
- 不要在本地重建模拟器/训练环境；不要改 `client/` 以外的服务器代码。

## 7. 回传

把 E-0 的 `summary.json`、`health.json`、`result.json`、`actions.jsonl`（可脱敏）和
结果屏截图回传服务器；大文件走 tar.gz + sha256，不进 Git。服务器据此判断是否进入 E-1。

## 8. 下一轮长训

本次**不启动**。服务器已准备 `config/league-stage2.json`、
`deploy/slurm/league_stage2.sbatch`、`artifacts/models/stage2-proposal-20261111/`
（p75band + forbid-unfreeze + stage fork），等实机准备完成后再讨论启动。
