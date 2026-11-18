import copy
import pytest
from workflows.probe_economy_suffix import compare

def rows():
    return [{'case':case,'source_seed':case,'variant':'normal','replica':replica,
        'future_seed':100+case*10+replica,'name':name,'rank':5+(case+1 if name=='student' else 0),
        'initial_sha256':str(case)+'-'+str(replica),
        'turns':[{'turn':t,'strength':100*(t-8),'gold_spent':15,
                  'forced_reason':None} for t in (9,12)]}
        for case in range(2) for replica in range(2) for name in ('parent','student')]

def test_suffix_comparison_clusters_replicates_by_source():
    result=compare(rows());r=result['paired']['student/normal']['rank_minus_parent']
    assert r['sources']==2 and r['paired_replicates']==4 and r['mean']==1.5
    assert result['paired']['student/normal']['common_survivor_T9_T12_log_growth_minus_parent']['mean']==0
    assert result['curves']['parent/normal']['turns']['12']['n']==4
    assert result['curves']['parent/normal']['turns']['12']['sources']==2

def test_suffix_comparison_rejects_unpaired_or_changed_seed():
    data=rows()
    with pytest.raises(ValueError,match='unpaired'):compare(data[:-1])
    data=copy.deepcopy(data);data[-1]['future_seed']+=1
    with pytest.raises(ValueError,match='future seed'):compare(data)
    data=rows();data[-1]['initial_sha256']='different'
    with pytest.raises(ValueError,match='initial state'):compare(data)


def test_supplementary_outcomes_keep_pairing_and_stable_intervals():
    from tools.suffix_outcomes import summarize
    data=rows()
    for r in data:r['rank']=1 if r['name']=='student' else 5
    result=summarize(data)
    win=result['paired']['student-parent/normal']['win_rate']
    assert win=={'mean':1.0,'ci95':[1.0,1.0],'sources':2,'paired_replicates':4}
    assert result['summary']['student/normal']['rank_counts']=={1:4}
    added=[{**r,'name':'third'} for r in data if r['name']=='parent']
    assert summarize(data+added)['paired']['student-parent/normal']==result['paired']['student-parent/normal']
    with pytest.raises(ValueError,match='unpaired'):summarize(data[:-1])
    broken=copy.deepcopy(data);broken[0]['initial_sha256']='changed'
    with pytest.raises(ValueError,match='identity'):summarize(broken)
    broken=copy.deepcopy(data);broken[0]['rank']=0
    with pytest.raises(ValueError,match='invalid final rank'):summarize(broken)


def test_sealed_study_merge_rejects_identity_changes(tmp_path):
    import json
    from hsbg.runtime.artifacts import atomic_json,file_identity
    from tools.analyze_economy_suffix import merge_studies
    roots=[]
    for name in ('parent','student'):
        root=tmp_path/name;root.mkdir();roots.append(root)
        run={k:{'identity':'same'} for k in ('engine','action_contract','source_manifest','snapshot_files')}
        atomic_json(root/'run.json',run)
        rr=[r for r in rows() if r['name']==name]
        for row in rr:row['trace']={'identity':{'checkpoint_sha256':name}}
        atomic_json(root/'complete.json',{'valid':True,'probe':False,'run':file_identity(root/'run.json'),
            'models':{name:{'sha256':name}},'rollouts':len(rr),'rows':rr})
    merged,_,result=merge_studies(roots)
    assert len(merged)==8 and result['paired']['student/normal']['rank_minus_parent']['mean']==1.5
    selected,_,_=merge_studies(roots,['parent'])
    assert len(selected)==4 and {r['name'] for r in selected}=={'parent'}
    selected,_,_=merge_studies(roots,['parent','student'],['normal'])
    assert len(selected)==8
    with pytest.raises(ValueError,match='whole variant missing'):
        merge_studies(roots,None,['normal','missing'])
    with pytest.raises(ValueError,match='whole policy arm missing'):
        merge_studies(roots,['parent','missing'])
    path=roots[1]/'complete.json';changed=json.loads(path.read_text());changed['rows'][0]['source_seed']=999
    atomic_json(path,changed,replace=True)
    with pytest.raises(ValueError,match='source seed'):merge_studies(roots)
    changed['rows'][0]['source_seed']=0;changed['rows'][0]['trace']['identity']['checkpoint_sha256']='wrong'
    atomic_json(path,changed,replace=True)
    with pytest.raises(ValueError,match='model identity'):merge_studies(roots)
