from tools.research_resource_ledger import totals


def test_allocations_do_not_double_count_typed_and_generic_gpu_counts():
    rows=[{'State':'COMPLETED','ElapsedRaw':'1800','AllocCPUS':'8',
        'AllocTRES':'billing=16,cpu=8,gres/gpu:h100-47=2,gres/gpu=2,mem=32G'},
        {'State':'CANCELLED by 123','ElapsedRaw':'0','AllocCPUS':'8','AllocTRES':''},
        {'State':'RUNNING','ElapsedRaw':'3600','AllocCPUS':'4','AllocTRES':'cpu=4,mem=8G'}]
    result=totals(rows)
    assert result['allocated_gpu_slice_hours']==1
    assert result['allocated_cpu_core_hours']==8
    assert result['states']=={'COMPLETED':1,'CANCELLED':1,'RUNNING':1}
