"""教师 DAgger 损失的归一化口径（审计 P2-1）。

背景：策略项按 **本微批合法行数** 求均值（``weight=valid``），而教师项按 **有标签行数**
求均值。两者分母不同 ⇒ 教师项的相对强度 = ``coef · N / n_t``，随每次采样的标签数
漂移（小批量时可达上百倍）。

本文件的测试钉住三件事：
1. 默认口径（``labeled``）与改动前**数值完全相同**（不改配方语义）；
2. 可选口径（``valid``）与策略项同分母，且等于 ``mean_labeled · n_t/N``；
3. 两种口径都会把 ``teacher_label_share = n_t/N`` 写进 stats（可观测性）。
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hsbg.az.train import ppo as P  # noqa: E402
from hsbg.az.train.ppo import PPOConfig, PPOBatch, ppo_losses  # noqa: E402

N, A = 8, P.N_ACTIONS


def _batch(teacher: np.ndarray) -> PPOBatch:
    """构造一个最小可跑批次：前 3 行 forced（不进策略/教师），其余 5 行合法。"""
    forced = np.zeros(N, dtype=bool)
    forced[:3] = True
    masks = np.zeros((N, A), dtype=bool)
    masks[:, 0] = True
    return PPOBatch(
        features=np.zeros((N, P.OBS_FLOATS), dtype=np.float32),
        cards=np.ones((N, P.OBS_CARDS), dtype=np.int32),
        masks=masks,
        actions=np.zeros(N, dtype=np.int64),
        old_logprobs=np.zeros(N, dtype=np.float32),
        values=np.zeros(N, dtype=np.float32),
        rewards=np.zeros(N, dtype=np.float32),
        dones=np.zeros(N, dtype=np.float32),
        advantages=np.zeros(N, dtype=np.float32),
        returns=np.zeros(N, dtype=np.float32),
        seats=np.zeros(N, dtype=np.int32),
        game_ids=np.arange(N, dtype=np.int64),
        ranks=np.full(N, 4, dtype=np.int64),
        turns=np.ones(N, dtype=np.int32),
        target_pi=np.zeros((N, A), dtype=np.float32),
        has_pi=np.zeros(N, dtype=bool),
        episode_ids=np.arange(N, dtype=np.int32),
        forced=forced,
        teacher_action=teacher,
    )


class _Net(torch.nn.Module):
    """极小的 logits 生成器：只要求输出 [B, N_ACTIONS] / [B, 8] / [B]。"""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(A))

    def forward(self, f, c, m=None):
        b = f.shape[0]
        return (self.w.unsqueeze(0).expand(b, A),
                torch.zeros(b, 8), torch.zeros(b))


def _loss(teacher, *, mode="labeled", coef=0.1):
    model = _Net()
    cfg = PPOConfig(device="cpu", clip=0.2, entropy_coef=0.0, value_coef=0.0,
                    pi_coef=0.0, seq_coef=0.0, teacher_bc_coef=coef,
                    teacher_loss_normalize=mode, normalize_advantage=False,
                    value_loss="mse")
    return ppo_losses(model, _batch(teacher), cfg)


def test_default_mode_is_labeled_and_matches_manual_mean():
    teacher = np.array([-1, -1, -1, 3, 3, -1, 3, -1], dtype=np.int16)
    out = _loss(teacher, mode="labeled")
    # 手动复算：只在 3 个有标签行上取 CE 平均（logits 全零 ⇒ CE = ln(245)）
    logits = torch.zeros(3, A)
    ta = torch.tensor([3, 3, 3])
    manual = torch.nn.functional.cross_entropy(logits, ta)
    assert float(out["teacher_bc_loss"]) == pytest.approx(float(manual), rel=1e-6)
    # 3 个标签 / 5 个合法行
    assert float(out["teacher_label_share"]) == pytest.approx(3 / 5)


def test_valid_mode_normalises_like_the_policy_term():
    teacher = np.array([-1, -1, -1, 3, 3, -1, 3, -1], dtype=np.int16)
    lab = float(_loss(teacher, mode="labeled")["teacher_bc_loss"])
    val = float(_loss(teacher, mode="valid")["teacher_bc_loss"])
    share = float(_loss(teacher, mode="valid")["teacher_label_share"])
    # valid 口径 = 求和 / 合法行数 = mean_labeled × n_t/N
    assert val == pytest.approx(lab * share, rel=1e-5)


def test_no_labels_means_no_teacher_term():
    teacher = np.full(N, -1, dtype=np.int16)
    out = _loss(teacher)
    assert "teacher_bc_loss" not in out
    assert "teacher_label_share" not in out


def test_forced_rows_are_excluded_from_labels():
    """forced 行即使带标签也不该计入（与策略项同一排除口径）。"""
    teacher = np.array([3, 3, 3, -1, -1, -1, -1, -1], dtype=np.int16)
    out = _loss(teacher)
    assert "teacher_bc_loss" not in out       # 全部落在 forced 行上 → 无标签


def test_invalid_normalize_mode_is_rejected():
    with pytest.raises(ValueError, match="teacher_loss_normalize"):
        PPOConfig(teacher_loss_normalize="mean")   # dataclass 在构造期就校验
