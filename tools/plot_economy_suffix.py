"""Render sealed paired suffix results without resampling or altering outcomes."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from hsbg.runtime.artifacts import atomic_json,file_identity

LABELS={'parent':'Parent v1.4-r1','operator':'Old heuristic','student':'Old BC student',
        'target_safe':'Target-safe heuristic','growth':'Growth heuristic'}
COLORS={'parent':'#52525b','operator':'#D55E00','student':'#0072B2',
        'target_safe':'#CC79A7','growth':'#009E73'}

def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    data=json.loads(Path(args.input).read_text());comparison=data['comparison']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(15,10.5),gridspec_kw={'width_ratios':[1.3,1,1]})
    arms=['operator','student','target_safe','growth']
    for row,variant in enumerate(('normal','complete_gold')):
        ax=axes[row,0]
        count_lines=['Surviving n at T8 / T9 / T10 / T11 / T12 / T13 / T14:']
        for name in ('parent','student','growth'):
            curve=comparison['curves'][name+'/'+variant]['turns']
            turns=[t for t in range(8,15) if str(t) in curve]
            ax.plot(turns,[curve[str(t)]['strength_median'] for t in turns],
                label=LABELS[name],color=COLORS[name],marker='o',markersize=3,
                linestyle='--' if name in ('operator','target_safe') else '-',linewidth=1.7)
            count_lines.append(LABELS[name]+': '+' / '.join(str(curve[str(t)]['n']) for t in turns))
        ax.set_xlabel('Turn');ax.set_ylabel('Survivor median attack + health');ax.grid(alpha=.2)
        ax.set_title('Normal three-core start' if variant=='normal' else 'Golden multiplier start',loc='left',fontweight='bold')
        if row==0:ax.legend(fontsize=8,loc='upper left')
        ax.text(0,-.22,'\n'.join(count_lines),transform=ax.transAxes,fontsize=7.5,va='top')
        for col,metric,title in ((1,'rank_minus_parent','Paired rank difference (lower is better)'),
            (2,'common_survivor_T9_T12_log_growth_minus_parent','Paired T9-T12 log-growth difference')):
            axis=axes[row,col]
            for i,name in enumerate(arms):
                estimate=comparison['paired'][name+'/'+variant][metric]
                mean=estimate['mean'];lo,hi=estimate['ci95']
                axis.errorbar(mean,i,xerr=np.array([[mean-lo],[hi-mean]]),fmt='o',
                    color=COLORS[name],capsize=3,linewidth=1.8)
                if col==2:axis.annotate('n='+str(estimate['sources']),(mean,i),xytext=(0,10),
                    textcoords='offset points',ha='center',fontsize=8,color=COLORS[name])
            axis.axvline(0,color='#9ca3af',linestyle='--',linewidth=1)
            axis.set_yticks(range(len(arms)),[LABELS[n] for n in arms]);axis.invert_yaxis()
            axis.set_ylim(len(arms)-.5,-.7);axis.grid(axis='x',alpha=.2);axis.set_title(title,fontsize=10)
            axis.set_xlabel('Difference from parent (95% CI)')
    fig.suptitle('Conditional economic skills: first-turn growth is not enough',fontsize=16,fontweight='bold',y=.985)
    fig.text(.02,.025,'49 source games x 2 future RNG replicas per arm and start type. Median curves use surviving replicas, not fixed cohorts.\n'
        'Growth contrasts use common T9/T12 survivors (n = independent source games). Explicit component endowments; not natural-game ability.',fontsize=9,color='#52525b')
    fig.tight_layout(rect=(0,.075,1,.955),w_pad=2.5,h_pad=4)
    fig.savefig(out/'conditional-payoff.png',dpi=180)
    fig.savefig(out/'conditional-payoff.svg')
    plt.close(fig)
    atomic_json(out/'manifest.json',{'input':file_identity(args.input),
        'outputs':{n:file_identity(out/n) for n in ('conditional-payoff.png','conditional-payoff.svg')},
        'scope':'Rendering of sealed descriptive comparisons, no new experiments or resampling'})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True)
    p.add_argument('--out',required=True);main(p.parse_args())
