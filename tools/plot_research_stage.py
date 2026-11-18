"""Plot complete sealed research comparisons, without new sampling or filtering."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from hsbg.runtime.artifacts import atomic_json,file_identity

LABELS={'parent':'Reference v1.4-r1','growth':'Growth teacher','query_replay':'Replay student',
        'query_aggregate':'Aggregated student','event_student':'Event student','payoff_student':'Phase student',
        'residual_no_history':'Residual: no history','residual_history':'Residual: public history'}
COLORS={'parent':'#52525b','growth':'#009E73','query_replay':'#0072B2','query_aggregate':'#CC79A7',
        'event_student':'#b45309','payoff_student':'#9ca3af','residual_no_history':'#E69F00','residual_history':'#7e22ce'}


def main(args):
    data=json.loads(Path(args.input).read_text());c=data['comparison'];variant=args.variant
    if not data['quality']['passed']:raise ValueError('admitted complete analysis required')
    names=args.policies
    if names[0]!='parent' or len(names)!=len(set(names)):raise ValueError('unique arms beginning with parent required')
    if any(n+'/'+variant not in c['curves'] for n in names):raise ValueError('missing complete arm')
    cohort=data['outcomes']['summary']['parent/'+variant]
    games,sources=cohort['games'],cohort['sources']
    if games%sources:raise ValueError('plot requires equal future replicas per source')
    replicas=games//sources
    fresh=data.get('source_scope',{}).get('role')=='assessment_only'
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(13.8,10))
    for ax,key,title in ((axes[0,0],'rank_minus_parent','Final rank difference: lower is better'),
            (axes[0,1],'common_survivor_T9_T12_log_growth_minus_parent','T9-T12 log-growth difference: higher is better')):
        for i,n in enumerate(names[1:]):
            e=c['paired'][n+'/'+variant][key];lo,hi=e['ci95'];mean=e['mean']
            ax.errorbar(mean,i,xerr=np.asarray([[mean-lo],[hi-mean]]),fmt='o',color=COLORS[n],capsize=4)
            ax.annotate(f"{e['sources']} sources / {e['paired_replicates']} pairs",(mean,i),xytext=(0,10),
                textcoords='offset points',ha='center',fontsize=8)
        ax.axvline(0,color='#9ca3af',linestyle='--');ax.grid(axis='x',alpha=.2)
        ax.set_yticks(range(len(names)-1),[LABELS[n] for n in names[1:]])
        ax.set_ylim(len(names)-1.5,-.7);ax.set_title(title,fontsize=11);ax.set_xlabel('Difference from reference, source-bootstrap 95% interval')
    for n in names:
        curve=c['curves'][n+'/'+variant]['turns'];turns=[t for t in range(8,15) if str(t) in curve]
        axes[1,0].plot(turns,[curve[str(t)]['strength_median'] for t in turns],color=COLORS[n],marker='o',markersize=3,label=LABELS[n])
        axes[1,1].plot(turns,[curve[str(t)]['n'] for t in turns],color=COLORS[n],marker='o',markersize=3,label=LABELS[n])
    axes[1,0].set_title('Survivor median total attack + health (P50)',fontsize=11)
    axes[1,1].set_title('Number of surviving future-RNG replicas',fontsize=11)
    axes[1,0].set_ylabel('Total attack + health');axes[1,1].set_ylabel(f'n (maximum {games}; not independent games)')
    for ax in axes[1]:ax.set_xlabel('Turn');ax.set_xticks(range(8,15));ax.grid(alpha=.2)
    axes[1,0].legend(fontsize=8,loc='upper left');axes[1,1].set_ylim(0,games*1.06)
    fig.suptitle('Complete-suffix assessment: fixed gold-engine starts',fontweight='bold',fontsize=16)
    cohort_label='Fresh assessment-only sources' if fresh else 'Exploratory development sources'
    fig.text(.02,.025,f'{sources} source games x {replicas} future RNG replicas per arm. Full final-rank pairs; growth uses common T9/T12 survivors.\n'
        f'{cohort_label}. Changing survivor cohorts; artificial components; no natural-game promotion claim.',fontsize=9,color='#52525b')
    fig.tight_layout(rect=(0,.085,1,.95),h_pad=2.5,w_pad=3)
    for ext in ('png','svg'):fig.savefig(out/('stage-results.'+ext),dpi=180)
    plt.close(fig)
    atomic_json(out/'manifest.json',{'input':file_identity(args.input),'policies':names,'variant':variant,
        'outputs':{p.name:file_identity(p) for p in out.iterdir()},'scope':'Rendering sealed estimates; no new resampling or outcome selection'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--policies',nargs='+',required=True);p.add_argument('--variant',default='complete_gold');main(p.parse_args())
