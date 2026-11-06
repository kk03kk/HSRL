"""验证稳健统计和回放等权口径，不依赖网络或原始数据。"""
from tools.analyze_firestone_strength import quantile,trimmed

def test_median_resists_large_tail():
 assert quantile([1,2,10**9],[1,1,1],.5)==2

def test_replay_weights_differ_from_board_count_weights():
 # 同回放两帧各半，另一回放一帧占1；两个回放各占一半。
 assert quantile([0,0,100],[.5,.5,1],.5)==50

def test_zero_board_and_even_sample_median():
 assert quantile([0,10],[1,1],.5)==5

def test_trimmed_mean_removes_tail_weight_not_raw_files():
 values=list(range(1,10))+[10**9]
 assert trimmed(values,[1]*10)==5.5
 assert values[-1]==10**9
