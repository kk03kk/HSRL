import numpy as np
from tools.audit_behavior_observability import audit


def test_exact_aliases_distinguish_source_splits_and_augmented_features():
    features=np.zeros((4,3282),np.float32);features[1,3227]=1
    mask=np.ones((4,245),bool);target=np.zeros_like(mask)
    target[np.arange(4),[1,8,1,8]]=True
    data={'features':features,'cards':np.zeros((4,124),np.int16),'masks':mask,'targets':target,
          'seeds':np.array([11,12,13,15]),'dev':np.array([False,False,False,True]),
          'kinds':np.array(['normal']*4)}
    old=audit(data,3227);augmented=audit(data,3282)
    assert old['train']['minimum_deterministic_label_errors']==1
    assert old['all']['minimum_deterministic_label_errors']==2
    assert old['dev']['conflicting_action_groups']==0
    assert old['train_dev_identical_observation_groups']==1
    assert augmented['train']['minimum_deterministic_label_errors']==0
    assert augmented['all']['minimum_deterministic_label_errors']==1
