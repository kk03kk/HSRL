import pytest
torch = pytest.importorskip("torch")

from hsbg.az.train.peer_actor import retain_material_mask
from hsbg.env.obs_std import GLOBAL_FIELD_INDEX, GLOBAL_FIELD_SCALE, MINION_F, SLICE_BOARD


@pytest.mark.parametrize("turn,hp,armor,units,remaining", [
    (7, 10, 5, [15, 35, 50], []),
    (7, 10, 5, [14, 36, 50], [0]),
    (6, 10, 5, [15, 35, 50], [0,1,2]),
    (7, 11, 5, [15, 35, 50], [0,1,2]),
    (8, 15, 0, [40,20,20,10,10], [0,1,2,3,4]),
    (12, 15, 0, [1500,8500], []),
])
def test_material_thresholds(turn, hp, armor, units, remaining):
    features = torch.zeros(1,3227)
    for name,value in dict(turn=turn,health=hp,armor=armor,board_len=len(units)).items():
        features[0,GLOBAL_FIELD_INDEX[name]] = value/GLOBAL_FIELD_SCALE[name]
    board = features[:,SLICE_BOARD].reshape(7,MINION_F)
    for i,stat in enumerate(units):
        board[i,:3] = torch.tensor([1,(stat-1)/10,.1])
    mask = torch.zeros(1,245,dtype=torch.bool)
    mask[0,0] = True
    mask[0,16:16+len(units)] = True
    result = retain_material_mask(features,mask)
    assert result[0,16:23].nonzero().flatten().tolist() == remaining
    assert result[0,0] and mask.sum() == 1+len(units)
    # A policy-level preference must leave at least one native action.
    mask[:] = False
    mask[0,16] = True
    assert torch.equal(retain_material_mask(features,mask),mask)
