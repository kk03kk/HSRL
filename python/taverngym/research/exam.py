"""不同合法脚本分布下的独立考试；不把profile名字当成已知棋力等级。"""
from dataclasses import replace

from taverngym.rl.train import evaluate


def exam(cfg, model, *, games=8):
    result = {}
    for profile in ("native", "tempo", "economy", "mixed"):
        c = replace(cfg, bot_profile=profile, opponent_mode="bots", eval_games=games,
                    eval_seeds=[1940000000, 1941000000])
        result[profile] = evaluate(c, model)
    return {"format": "taverngym.distribution-exam.v1", "human_calibrated": False,
            "protocol": "held-out seeds; seven legal scripted opponents; profile strength unproven",
            "results": result}
