"""流水线旧批次零更新时，保持预采集结果并切换本进程同步采样。"""


def after_update(enabled,steps,lag,executor,allow_fallback):
    if allow_fallback and enabled and lag>0 and steps==0:
        # 当前没有optimizer.step，参数未变；已发出的下一批仍来自当前参数。
        # 等它完成，调用方保留Future供下一轮消费；不丢弃或重标行为logprob。
        if executor is not None:executor.shutdown(wait=True)
        return False,None,True
    return enabled,executor,False
