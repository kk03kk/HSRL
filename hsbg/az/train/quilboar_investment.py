"""Price the entry cost, permanent gem size and available distribution."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from hsbg.env import obs_std as O
from .quilboar_option import QuilboarOption, TERMS, basis, roles


@lru_cache(None)
def support():
    cards = json.loads((Path(__file__).resolve().parents[3] /
                        "engine/crates/hsbg-engine/data/cards.json").read_text(encoding="utf-8"))["cards"]
    ids = [None] + [c["normal"] or c["id"] for c in cards]
    return np.array(ids, dtype=object)


def investment(f, c):
    if f[O.GLOBAL_LOBBY_RACES_START + 7] < .5 or f[0] < .19:
        return None
    slices = (O.SLICE_CARD_BOARD, O.SLICE_CARD_HAND, O.SLICE_CARD_SHOP, O.SLICE_CARD_CHOICES)
    b,h,s,d = (support()[c[sl]] for sl in slices)
    br,hr,sr,dr = (roles()[0][c[sl]] for sl in slices)
    pending = support()[c[O.SLICE_CARD_PENDING_SOURCE][0]]
    growth = ("BG31_326","BG33_885","BG23_017","BG30_123","BG31_320")
    delivery = ("BG34_684","BG28_583","BG36_341","BG31_323","BG30_121")
    def owns(ids):
        return np.isin(b,ids).sum() + np.isin(h,ids).sum()
    owned_core = int(br[:,:2].sum() + hr[:,:2].sum())
    boars = int(br[:,3].sum())
    # Limit speculative entry. Standalone support cards first need another
    # source of public evidence, such as two potential recipients.
    if not owned_core and not any(v[:,:2].any() for v in (sr,dr)):
        if boars < 2 or not any(np.isin(v,growth+delivery).any() for v in (b,h,s,d)):
            return None
    gem = float(5 * (f[31] + f[32]))
    pieces = f[O.SLICE_BOARD].reshape(7,O.MINION_F)
    mass = 10 * (pieces[:,O.MINION_ATK] + pieces[:,O.MINION_HP])
    health = f[5]*40 + f[6]*20
    ready = min(1., .16*boars + .07*gem + .2*(owned_core>0))
    cost = min(1., mass.sum()/650 + max(0.,22-health)/28) * (1-ready)
    anchors = basis(f,c)
    x = np.zeros((245,2*len(TERMS)+2),np.float32)
    if anchors is not None:
        x[:,:len(TERMS)] = anchors
        x[:,len(TERMS):2*len(TERMS)] = -anchors * cost
    # A gold spent growing gems pays off more with existing distributors.
    # Recruiting a distributor pays off more when gems have already grown.
    invest = min(1.,(.48+.15*min(3,owns(delivery))+ .10*(boars>=2)) *
                 (1-.65*cost)) * max(.35,1-min(.65,gem/42))
    harvest = min(1.6,.20+.105*gem+.14*min(3,owns(growth))) * (1-.5*cost)
    def values(ids):
        return (np.isin(ids,growth)*invest, np.isin(ids,delivery)*harvest)
    def options(dst,ids):
        size,pay = values(ids)
        # Already-owned permanent utility pieces have diminishing marginal value.
        other = np.array([.45 if k in b or k in h else 1. for k in ids])
        dst[:,-2],dst[:,-1] = size*other,pay*other
    options(x[8:16],s)
    if f[O.SLICE_PENDING][2]>.5:
        options(x[30:46],d)
    prod,pay = values(h)
    play = x[46:126].reshape(10,8,-1)
    play[:,:,-2] = (prod + np.isin(h,("BG31_893",))*invest)[:,None]
    play[:,:,-1] = (pay + np.isin(h,("BG34_689","BG28_698"))*harvest)[:,None]
    # The native Choose One pending source persists after the hand card leaves.
    if f[O.SLICE_PENDING][3]>.5:
        if pending=="BG30_123":
            x[30,-2] += invest
        elif pending=="BG31_320":
            x[31,-2] += invest
        elif pending=="BG31_893":
            x[30 if f[31]<f[32] else 31,-2] += .5*invest
    bg,bp = values(b)
    x[16:23,-2] -= .65*bg
    x[16:23,-1] -= .7*bp
    # Create space only when the gained public piece can use it. Pricing
    # the actual body lost matters on a developed dragon or mech board.
    if br[:,5].sum()==7 and np.any((prod+pay)>0):
        replace = np.maximum(0,1-mass/(30+mass.sum()/10))**2
        x[16:23,-1] += replace*(~br[:,:2].any(1))*(1-.7*(bg+bp>0))*harvest
    return x


class QuilboarInvestment(QuilboarOption):
    arch = "quilboar_investment_v1"
    terms = TERMS + tuple("entry_cost_" + k for k in TERMS) + ("size","delivery")
    features = staticmethod(investment)
