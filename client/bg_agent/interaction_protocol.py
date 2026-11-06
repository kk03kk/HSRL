"""Shared pre-input validation for server-described sequential interactions."""


def branch_entities(action):
    stages=action.interaction.stages if action.interaction else ()
    options=tuple(s.target for s in stages if s.mode=='branch' and s.operation=='candidate')
    selected=[s.target for s in stages if s.mode=='branch' and s.operation=='choose']
    if action.sub_option_id is None or not 1<=len(options)<=10 or len(set(options))!=len(options):
        raise ValueError('branch_selection_mapping_unknown')
    if any(not isinstance(e,int) or isinstance(e,bool) or e<=0 for e in options) or len(selected)!=1 or selected[0] not in options:
        raise ValueError('branch_selection_mapping_unknown')
    if any(not s.requires_new_observation or s.timeout_ms<=0 for s in stages if s.mode=='branch'):
        raise ValueError('branch_requires_fresh_observation')
    return options,selected[0]


def quest_reward_links(obs):
    """Pure entity association, shared by perception and portable policies."""
    rewards={c.entity_id:c for c in obs.choice_rewards}
    if not obs.choices:return {}
    links={}
    for c in obs.choices:
        try:eid=int(c.tags.get('TAG_SCRIPT_DATA_ENT_1',''))
        except (ValueError,TypeError):return {}
        reward=rewards.get(eid)
        if c.tags.get('QUEST')!='1' or not reward or reward.card_type!='BATTLEGROUND_QUEST_REWARD':return {}
        links[c.entity_id]=reward
    return links


