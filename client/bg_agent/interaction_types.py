"""Policy-independent interaction vocabulary; no card-specific gestures.

Families are composable: a targeted Choose One spell is PLAY + BRANCH + TARGET.
The catalogue is an extensible coverage checklist, never a claim of a complete
finite list of all current/future Hearthstone interfaces or of live validation.
"""
from dataclasses import asdict, dataclass
from .model import Kind, Phase


@dataclass(frozen=True)
class InteractionType:
    id: str
    title: str
    implementation: str  # implemented / partial / capture_only / out_of_scope
    acceptance: str


TYPES = tuple(InteractionType(*row) for row in (
    ('enter', '开始单人对局', 'implemented', '新对局建立且进入英雄选择/招募'),
    ('matchmaking', '排队与开局等待', 'partial', '不重复排队；失败/超时需显式诊断'),
    ('choice_hero', '英雄选择及确认', 'partial', '确认前选中身份；确认后本地英雄身份'),
    ('hero_reroll', '英雄重掷/代币入口', 'capture_only', '独立授权与费用；当前禁止自动操作'),
    ('buy', '购买随从或酒馆法术', 'implemented', '源实体控制者/区域转移；三连消耗需因果链'),
    ('sell', '出售场上随从', 'implemented', '匹配回执及指定实体离场'),
    ('upgrade', '酒馆升级', 'implemented', '匹配回执及本数增加'),
    ('refresh', '刷新商店', 'implemented', '匹配回执及商店实体集合改变'),
    ('freeze', '冻结/解冻切换', 'implemented', '同一商店集合冻结标签一致切换'),
    ('play_minion', '随从入场/战吼起始动作', 'implemented', '指定实体入场；即死需因果区域链'),
    ('play_spell', '普通/酒馆/临时法术', 'implemented', '匹配回执、来源消耗及强制阶段关闭'),
    ('insertion', '指定插入位置', 'partial', '完整位置几何及最终实际位置'),
    ('magnetize', '磁力合体', 'partial', '已知磁力语义、指定目标及合体属性证据'),
    ('reorder_board', '己方场面换位', 'implemented', '最终实体全序列匹配'),
    ('reorder_shop', '商店换位', 'capture_only', '需要专用商店几何/核验；生产未放行'),
    ('hero_power', '主动英雄技能', 'partial', '费用/耗竭/进度或可观察效果，不接受裸回执'),
    ('activate', '发动随从/专用机制按钮', 'partial', '控件槽位和协议有证据；能力使用证明'),
    ('target', '指定己方/商店/英雄等目标', 'partial', '当前合法目标、实时高亮、匹配结果'),
    ('target_hand', '手牌目标', 'capture_only', '手牌目标高亮/遮挡协议尚无真实验证'),
    ('target_opponent', '对手目标', 'capture_only', '对手实体和目标区域尚未纳入生产接口'),
    ('target_unknown', '未投影目标', 'capture_only', '不能将未知实体擅自标成对手目标'),
    ('branch', 'Options 抉择分支', 'partial', '完整分支顺序、合法分支及新鲜可点击证据'),
    ('branch_target', '先抉择再指定目标', 'partial', '同一父动作的分支→目标因果顺序'),
    ('choice_card', '卡牌发现/三连奖励选择', 'implemented', '请求回执及奖励/关闭证据；奖励身份分级'),
    ('choice_effect', '效果/路径/机制选择', 'partial', '原请求提交并关闭；不假定奖励一定入手'),
    ('choice_power', '选择英雄技能', 'partial', '候选顺序及请求回执；主动能力身份另核对'),
    ('choice_trinket', '购买饰品并确认', 'partial', '价格、选中槽位、确认及激活身份'),
    ('choice_quest', '任务与奖励成对选择', 'partial', '完整任务/奖励日志关联、专用面板布局及请求结果核验'),
    ('choice_multi', '多选集合再确认', 'partial', '每项选中态、数量限制、完整服务器集合'),
    ('choice_optional', '可跳过/零选项提交', 'capture_only', 'CountMin=0 不等于已有取消/跳过控件'),
    ('choice_chain', '连续/嵌套强制选择', 'partial', '父子请求关联及各自关闭，不混用回执'),
    ('confirm', '已选集合/选项确认', 'partial', '确认前所选身份、请求及启用态一致'),
    ('cancel', '取消目标/返回选择', 'capture_only', '明确可取消协议；不能通用 Escape 猜测'),
    ('grid_paged', '网格/分页/滚动选择', 'capture_only', '需要独立槽位/页标识；单排适配器不得放行'),
    ('combat_wait', '战斗/动画/回合切换等待', 'implemented', '日志阶段与客户端场景分离；有界等待'),
    ('result', '结算及返回菜单', 'partial', '本地终局证据及真实菜单返回'),
    ('pause_menu', '暂停菜单/遮挡模态', 'partial', '只识别并阻断输入；不自动关闭'),
    ('reconnect', '断线/恢复/投影重建', 'partial', '只读对账；不确定阶段禁止重发'),
    ('ui_unknown', '未知界面与未映射授权', 'capture_only', '留证并拒绝，不生成坐标'),
    ('duos', '双人传递/传送门/队友选择', 'out_of_scope', '当前单人模式拒绝；单独记录范围'),
))
BY_ID = {row.id: row for row in TYPES}


def action_types(action, obs):
    """Describe protocol structure, not UI readiness or card effect rules."""
    k = action.kind
    base = {Kind.ENTER: 'enter', Kind.BUY: 'buy', Kind.SELL: 'sell',
        Kind.UPGRADE: 'upgrade', Kind.REFRESH: 'refresh', Kind.FREEZE: 'freeze',
        Kind.HERO_POWER: 'hero_power', Kind.ACTIVATE: 'activate',
        Kind.TARGET: 'target', Kind.CONFIRM: 'confirm', Kind.CANCEL: 'cancel',
        Kind.DISMISS: 'result'}.get(k)
    result = [base] if base else []
    cards = {c.entity_id: c for c in obs.hand + obs.board + obs.shop + obs.choices + obs.controls}
    if k == Kind.PLAY:
        card = cards.get(action.source)
        result.append('play_minion' if card and card.card_type == 'MINION' else 'play_spell' if card and card.card_type in ('SPELL','BATTLEGROUND_SPELL') else 'ui_unknown')
        if action.position is not None: result.append('insertion')
        if action.interaction and action.interaction.placement == 'magnetize': result.append('magnetize')
    if k == Kind.REORDER:
        result.append('reorder_shop' if action.scope == 'shop' else 'reorder_board')
    if k in (Kind.CHOOSE, Kind.SELECT):
        types = {c.card_type for c in obs.choices}
        if types == {'HERO'}: result.append('choice_hero')
        elif types == {'BATTLEGROUND_TRINKET'}: result.append('choice_trinket')
        elif types == {'HERO_POWER'}: result.append('choice_power')
        elif any(c.tags.get('QUEST') == '1' for c in obs.choices): result.append('choice_quest')
        elif types <= {'MINION', 'SPELL', 'BATTLEGROUND_SPELL'}: result.append('choice_card')
        else: result.append('choice_effect')
        if obs.interaction and obs.interaction.max_select > 1: result.append('choice_multi')
        if obs.interaction and obs.interaction.min_select == 0: result.append('choice_optional')
    if action.sub_option_id is not None:
        result.append('branch')
        if action.target is not None: result.append('branch_target')
    if action.target is not None and not (action.interaction and action.interaction.placement == 'magnetize'):
        if 'target' not in result: result.append('target')
        if action.target in {c.entity_id for c in obs.hand}: result.append('target_hand')
        known = set(cards) | ({obs.hero.entity_id} if obs.hero else set())
        if action.target not in known: result.append('target_unknown')
    return tuple(dict.fromkeys(result))


def describe(obs):
    families = sorted({family for a in obs.actions for family in action_types(a, obs)})
    if obs.phase == Phase.COMBAT: families.append('combat_wait')
    if obs.phase == Phase.GAME_OVER: families.append('result')
    ctx = obs.interaction
    return dict(schema=1, families=sorted(set(families)), phase=str(obs.phase),
        choice_count=len(obs.choices), choice_types=sorted({c.card_type for c in obs.choices}),
        bounds=[ctx.min_select, ctx.max_select] if ctx else None,
        hand_count=len(obs.hand), board_count=len(obs.board),
        branch_sizes=sorted({sum(s.mode == 'branch' and s.operation == 'candidate'
            for s in a.interaction.stages) for a in obs.actions if a.interaction and a.sub_option_id is not None}),
        evidence='semantic structure only; layout and live execution unproven')


def catalogue():
    return [asdict(row) for row in TYPES]
