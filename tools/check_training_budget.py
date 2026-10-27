"""Offline adversarial Move-loop probe; no game client, training or input control."""
import json
from pathlib import Path
import types
from local_offline import prepare


def main():
    handles = prepare()
    import torch
    import taverngym as tg
    from taverngym.rl.config import Config, TRAIN_ACTION_BUDGET
    from taverngym.rl.collect import collect
    from taverngym.rl.model import Policy
    cfg = Config(device='cpu', policy_kind='compact_ff', opponent_mode='bots',
                 n_games=1, learner_seats=1, threads=1, torch_threads=1,
                 dim=16, layers=1, heads=2, hidden=16, seq_len=8, chunks_per_batch=2)
    torch.set_num_threads(1); torch.manual_seed(19)
    model = Policy(cfg)
    original = model.forward
    actions = json.loads(tg.Game(1).view(0))['actions']

    def looping(self, obs, hidden):
        logits, value, rank, nh = original(obs, hidden)
        logits = logits.clone()
        for start, end, score in [
            (actions['BUY'], actions['SELL'], 200.),
            (actions['PLAY'], actions['MAGNETIZE'], 400.),
            (actions['MOVE'], tg.N_ACTIONS, 1000.),
        ]:
            logits[:, start:end] = torch.where(
                obs['mask'][:, start:end], torch.full_like(logits[:, start:end], score),
                logits[:, start:end])
        return logits, value, rank, nh

    model.forward = types.MethodType(looping, model)
    trajectories, stats = collect(cfg, model, 900000000, greedy=True)
    hits = sum(s['action'] == 0 and s['obs']['extras'][0][2][1] >= TRAIN_ACTION_BUDGET
               for tr in trajectories for s in tr['samples'])
    moves = sum(s['action'] >= actions['MOVE'] for tr in trajectories for s in tr['samples'])
    result = {'purpose':'Synthetic loop probe; not model strength', 'budget':TRAIN_ACTION_BUDGET,
              'native':tg.build_info(), 'stats':stats, 'budget_end_turns':hits, 'moves':moves}
    assert hits > 0 and moves > 0 and stats['truncated_games'] == 0
    out = Path(__file__).resolve().parents[1]/'runs/training-budget-200'
    out.mkdir(parents=True, exist_ok=True)
    (out/'loop-probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
