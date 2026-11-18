import json
import pytest
from hsbg.runtime.artifacts import file_identity
from tools.analyze_economy_suffix import assessment_scope


def test_fresh_analysis_label_requires_bound_manifest_and_matching_sources(tmp_path):
    bank=tmp_path/'bank';bank.mkdir();root=tmp_path/'run';root.mkdir()
    manifest=bank/'manifest.json';manifest.write_text(json.dumps({'assessment_only':True,'seeds':[100,101,102]}))
    (root/'run.json').write_text(json.dumps({'source_manifest':file_identity(manifest)}))
    rows=[{'source_seed':100},{'source_seed':100},{'source_seed':102}];identities=[{'directory':str(root)}]
    r=assessment_scope(rows,identities,bank)
    assert r['evaluated_source_games']==2 and r['collected_source_games']==3
    with pytest.raises(ValueError,match='outside'):assessment_scope([{'source_seed':999}],identities,bank)
    manifest.write_text(json.dumps({'assessment_only':True,'seeds':[100,101,102,103]}))
    with pytest.raises(ValueError,match='identity differs'):assessment_scope(rows,identities,bank)
    manifest.write_text(json.dumps({'seeds':[100,101,102]}))
    with pytest.raises(ValueError,match='assessment-only'):assessment_scope(rows,identities,bank)
