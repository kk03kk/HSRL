"""模型架构开关（z-token / 卡表属性 / 磁力查询）的兼容性与正确性测试。

对应审计 P0-A1 / P0-A2 / E / 「z_global 是死代码」：
* 带这些开关训练出的 checkpoint 以前**无法再加载**（``strict=True`` 直接报
  unexpected/missing key），因为开关既不写进 meta，也没有加载端推断；
* ``z_global`` 算了却从未加进 global token（梯度恒 None）；
* ``z=None`` 的默认路径"等价旧模型"以前只是"两个 bug 抵消"的巧合。

全部测试**不需要 hsbg_engine**（用假 ``CardTables`` 注入）。
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hsbg.az.train import entity_model as EM  # noqa: E402
from hsbg.az.train import model as M  # noqa: E402

N_CARDS = 8
OBS_FLOATS = 3227
N_ACTIONS = 245


def _tables() -> "EM.CardTables":
    # ``card_text_proj`` 是固定宽度 Linear(TEXT_DIM, card_dim) ⇒ text 必须是
    # 恰好 ``EM.TEXT_DIM`` 列；``card_static`` 的列数随意（投影层按形状建）。
    return EM.CardTables(
        static=np.zeros((N_CARDS, EM.CARD_STATIC_DIM), dtype=np.float32),
        text=np.zeros((N_CARDS, EM.TEXT_DIM), dtype=np.float32),
        n_cards=N_CARDS)


@pytest.fixture()
def fake_tables(monkeypatch):
    """让 ``load_card_tables()`` 返回假表，从而在无引擎环境下建网/存读 ckpt。"""
    t = _tables()
    monkeypatch.setattr(EM, "load_card_tables", lambda *a, **k: t)
    return t


def _net(**kw):
    kw.setdefault("d_model", 32)
    kw.setdefault("n_layers", 1)
    kw.setdefault("n_heads", 4)
    kw.setdefault("tables", _tables())
    return EM.build_entity_net(**kw)


def _batch(B=2):
    f = torch.zeros(B, OBS_FLOATS)
    c = torch.arange(1, B * 124 + 1, dtype=torch.long).reshape(B, 124) % N_CARDS
    m = torch.zeros(B, N_ACTIONS, dtype=torch.bool)
    m[:, 0] = True
    return f, c, m


# ────────────────────────── 参数与 buffer 形状 ──────────────────────────
def test_extra_params_and_buffers_exist():
    base, full = _net(), _net(use_z=True, use_attr=True, mag_query=True)
    pb = sum(p.numel() for p in base.parameters())
    pf = sum(p.numel() for p in full.parameters())
    assert pf > pb
    assert tuple(full.zcard.shape) == (46, N_CARDS, 3)
    assert tuple(full.attr.shape) == (N_CARDS, 311)
    assert float(full.mag_q.weight.abs().sum()) == 0.0     # 零初始化 = 起点等价旧版
    # 新增 buffer 必须出现在 state_dict 里（否则 A1 的"键集不一致"无从谈起）
    keys = full.state_dict().keys()
    for k in ("zcard", "attr", "z_global.weight", "z_tok.weight",
              "attr_proj.weight", "mag_q.weight"):
        assert k in keys


def test_arch_flags_self_describe():
    flags = _net(use_z=True, use_attr=True, mag_query=True).arch_flags()
    assert flags == {"use_z": True, "n_z": 46, "use_attr": True,
                     "attr_f": 311, "mag_query": True}
    assert _net().arch_flags()["use_z"] is False


# ────────────────────────── 默认关 = 逐位等价旧模型 ──────────────────────────
def test_zero_buffers_make_new_features_exactly_noop():
    """零 buffer + 中立槽 ⇒ 带三个开关的模型与旧模型 logits **逐位相等**。"""
    base = _net()
    full = _net(use_z=True, use_attr=True, mag_query=True)
    full.load_state_dict(base.state_dict(), strict=False)   # 只载入公共权重
    f, c, m = _batch()
    base.eval(); full.eval()
    with torch.no_grad():
        lb, rb, vb = base(f, c, m)
        lf, rf, vf = full(f, c, m)          # z=None → 中立槽
    assert torch.equal(lb, lf)
    assert torch.equal(rb, rf) and torch.equal(vb, vf)


def test_scalar_z_agrees_with_vector_z():
    full = _net(use_z=True).eval()      # eval：关掉 dropout，两次前向才可比
    f, c, m = _batch()
    with torch.no_grad():
        a = full.forward_global(f, c, m, z=torch.tensor(6))
        b = full.forward_global(f, c, m, z=torch.full((2,), 6))
    assert torch.allclose(a, b)


# ────────────────────────── z_global 不再死代码 ──────────────────────────
def test_z_global_reaches_global_token():
    """回归：``_zg`` 以前算了但没加进 g ⇒ z=6 与 z=45 完全相同、梯度为 None。"""
    full = _net(use_z=True)
    f, c, m = _batch()
    with torch.no_grad():
        g6 = full.forward_global(f, c, m, z=torch.full((2,), 6))
        g45 = full.forward_global(f, c, m, z=torch.full((2,), 45))
    assert not torch.allclose(g6, g45)


def test_z_global_gets_gradient():
    full = _net(use_z=True)
    f, c, m = _batch()
    logits, _r, _v = full(f, c, m, z=torch.full((2,), 6))
    logits.sum().backward()
    assert full.z_global.weight.grad is not None
    assert float(full.z_global.weight.grad.abs().sum()) > 0.0
    # 中立列永远不该有梯度（它的 one-hot 输入恒为 0 时也不该被用到）
    assert float(full.z_global.weight.grad[:, 45].abs().sum()) == 0.0


def test_neutral_slot_is_zero_at_init():
    full = _net(use_z=True)
    assert float(full.zcard[45].abs().sum()) == 0.0
    assert float(full.z_global.weight[:, 45].abs().sum()) == 0.0


def test_zero_neutral_slot_restores_invariant_after_poisoning():
    """加载外来 zcard 之后必须再调 zero_neutral_slot（否则默认路径不再是旧模型）。"""
    full = _net(use_z=True)
    with torch.no_grad():
        full.zcard[45].fill_(1.0)
        full.z_global.weight[:, 45].fill_(0.5)
    info = full.zero_neutral_slot()
    assert info["zcard_neutral_abs_sum_before"] > 0
    assert float(full.zcard[45].abs().sum()) == 0.0
    assert float(full.z_global.weight[:, 45].abs().sum()) == 0.0


# ────────────────────────── checkpoint 往返（A1/A2）──────────────────────────
def test_roundtrip_with_all_switches(fake_tables, tmp_path):
    """带三个开关的模型必须能 save→load 回来，且 logits 逐位相同。"""
    full = _net(use_z=True, use_attr=True, mag_query=True)
    with torch.no_grad():                      # 让新增参数非零，才能验出"没被丢掉"
        full.mag_q.weight.normal_()
        full.z_tok.weight.normal_()
        full.attr_proj.weight.normal_()
        full.zcard[6].fill_(1.0)
    path = str(tmp_path / "m.pt")
    M.save_checkpoint(full, path, meta={"arch": "entity", "d_model": 32,
                                        "n_layers": 1, "n_heads": 4})
    back, meta = M.load_checkpoint(path, device="cpu")
    assert meta["use_z"] is True and meta["n_z"] == 46
    assert meta["use_attr"] is True and meta["attr_f"] == 311
    assert meta["mag_query"] is True
    f, c, m = _batch()
    full.eval(); back.eval()
    with torch.no_grad():
        a = full(f, c, m, z=torch.full((2,), 6))
        b = back(f, c, m, z=torch.full((2,), 6))
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_default_checkpoint_meta_has_flags_false(fake_tables, tmp_path):
    base = _net()
    path = str(tmp_path / "b.pt")
    M.save_checkpoint(base, path, meta={"arch": "entity", "d_model": 32,
                                        "n_layers": 1, "n_heads": 4})
    back, meta = M.load_checkpoint(path, device="cpu")
    assert meta["use_z"] is False and meta["use_attr"] is False
    assert meta["mag_query"] is False


def test_load_infers_switches_from_state_dict_when_meta_missing(fake_tables, tmp_path):
    """老 ckpt 没有 meta 开关（这里手工抹掉），也要靠键名推断出来。"""
    full = _net(use_z=True, use_attr=True, mag_query=True)
    path = str(tmp_path / "m.pt")
    M.save_checkpoint(full, path, meta={"arch": "entity", "d_model": 32,
                                        "n_layers": 1, "n_heads": 4})
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("use_z", "n_z", "use_attr", "attr_f", "mag_query"):
        payload["meta"].pop(k, None)
    torch.save(payload, path)
    back, meta = M.load_checkpoint(path, device="cpu")
    assert back.use_z is True and back.use_attr is True and back.mag_query is True
    assert back.n_z == 46


def test_loaded_model_neutral_slot_is_rezeroed(fake_tables, tmp_path):
    """外来 zcard 的第 45 槽即使非零，加载后也必须被清零。"""
    full = _net(use_z=True)
    path = str(tmp_path / "m.pt")
    M.save_checkpoint(full, path, meta={"arch": "entity", "d_model": 32,
                                        "n_layers": 1, "n_heads": 4})
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["state_dict"]["zcard"][45].fill_(3.0)
    payload["state_dict"]["z_global.weight"][:, 45].fill_(3.0)
    torch.save(payload, path)
    back, _meta = M.load_checkpoint(path, device="cpu")
    assert float(back.zcard[45].abs().sum()) == 0.0
    assert float(back.z_global.weight[:, 45].abs().sum()) == 0.0
