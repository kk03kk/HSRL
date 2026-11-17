import copy
import pytest
from workflows.probe_economy_suffix import compare

def rows():
    return [{'case':case,'source_seed':case,'variant':'normal','replica':replica,
        'future_seed':100+case*10+replica,'name':name,'rank':5+(case+1 if name=='student' else 0),
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
