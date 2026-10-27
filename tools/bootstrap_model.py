"""Run the supervised recovery stage using the project's existing environment."""
import argparse
from local_offline import prepare


def main():
    handles=prepare()
    from taverngym.research.bootstrap import bootstrap
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--init-from');p.add_argument('--waves',type=int,default=16)
    p.add_argument('--epochs',type=int,default=4);p.add_argument('--bc-lr',type=float,default=3e-4)
    p.add_argument('--batch-chunks',type=int,default=8);p.add_argument('--gate-games',type=int,default=8)
    a=p.parse_args()
    if min(a.waves,a.epochs,a.batch_chunks,a.gate_games)<1 or a.bc_lr<=0:p.error('positive bootstrap budgets required')
    r=bootstrap(a.config,a.out,init_from=a.init_from,waves=a.waves,epochs=a.epochs,
                bc_lr=a.bc_lr,batch_chunks=a.batch_chunks,gate_games=a.gate_games)
    return 0 if r['ready_for_ppo'] else 3


if __name__=='__main__':raise SystemExit(main())
