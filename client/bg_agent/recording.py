"""Lossless local log mirror and bounded, observational match diagnostics.

No game input. Raw logs stay on this computer; reports distinguish human
SendOption events from hypothetical policy decisions.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import re
import time

from .model import Kind
from .policy import RulePolicy
from .power import LINE, PowerState
from .verification import verify,Status
from . import __version__


def atomic_replace(source,target,attempts=4):
    """Windows readers may briefly hold a file without delete sharing."""
    for attempt in range(attempts):
        try:
            source.replace(target)
            return attempt
        except PermissionError:
            if attempt+1==attempts:raise
            time.sleep(.01*(2**attempt))


def compact_observation(observation,*,limit=24):
    """Bounded diagnostic copy: newest receipt identities, no post-state.

    The live observation keeps every causal snapshot for verification; audit
    and evidence copies only need the newest identities. Without this a single
    late-game observation row is ~300 KB and a full match rotates the audit
    away before it can be read.
    """
    resolutions=observation.resolutions[-limit:] if limit else observation.resolutions
    return replace(observation,resolutions=tuple(replace(r,post_state=None) if r.post_state is not None else r
                                                 for r in resolutions))


def compact_observation_dict(data,*,limit=24):
    """Same compaction for an already-serialized observation dict."""
    resolutions=(data.get('resolutions') or ())[-limit:] if limit else (data.get('resolutions') or ())
    return {**data,'resolutions':[{**r,'post_state':None} for r in resolutions]}


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    return atomic_replace(temporary,path)


class RawMirror:
    """Follows the same file identity after rename, never silently skips bytes."""
    def __init__(self, source: Path, destination: Path):
        self.source=source
        stat=source.stat()
        self.identity=(stat.st_dev,stat.st_ino)
        self.destination=destination
        self.offset=0
        self.digest=hashlib.sha256()
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.touch(exist_ok=False)

    def poll(self, *, max_bytes=None):
        if max_bytes is not None and max_bytes < 0:
            raise ValueError('max_bytes must be nonnegative')
        try: stat=self.source.stat()
        except FileNotFoundError: stat=None
        if stat is None or (stat.st_dev,stat.st_ino)!=self.identity:
            # Rotated logs commonly end in .log.1 rather than .log. Never read
            # a different file merely because its name matches: identity wins.
            for path in self.source.parent.iterdir():
                if not path.is_file():continue
                stat=path.stat()
                if (stat.st_dev,stat.st_ino)==self.identity:
                    self.source=path
                    break
            else:
                raise OSError("原日志已被替换且找不到原文件，不能保证完整采集")
        if self.source.stat().st_size<self.offset:
            raise OSError("原日志已截断，已保留副本；不能静默续写为完整日志")
        with self.source.open("rb") as source:
            source.seek(self.offset)
            chunk=source.read() if max_bytes is None else source.read(max_bytes)
        if chunk:
            with self.destination.open("ab") as target: target.write(chunk)
            self.offset+=len(chunk)
            self.digest.update(chunk)
        return chunk

    def manifest(self):
        return dict(source=str(self.source),file=self.destination.name,bytes=self.offset,sha256=self.digest.hexdigest())


class MatchAnalysis:
    def __init__(self, directory: Path, player_id=None):
        self.directory=directory
        self.state=PowerState(player_id)
        self.policy=RulePolicy()
        self.family=None
        self.last_phase=None
        self.turns=set()
        self.phases=Counter()
        self.decisions=Counter()
        self.human=Counter()
        self.gaps=Counter()
        self.choices=Counter()
        self.lines=0
        self.first_time=None
        self.last_time=None
        self.terminal_seen=False
        self.terminal_evidence=None
        self.events=0
        self.last_options_actions=()
        self.last_options_cards={}
        self.pending_actions=[]
        self.used_receipts=set()
        self.outcomes=Counter()
        self.choice_submitted=0
        self.choice_mapped=0
        self.choice_before=None
        self.choice_id=None

    def evaluate(self,observation=None):
        if not self.pending_actions:return
        obs=observation or self.state.observe()
        remaining=[]
        for request in self.pending_actions:
            available=replace(obs,resolutions=tuple(r for r in obs.resolutions if r.at_revision not in self.used_receipts))
            result=verify(request["action"],request["before"],available)
            if result.status==Status.PENDING:
                remaining.append(request);continue
            if result.receipt_revision is not None:self.used_receipts.add(result.receipt_revision)
            self.outcomes[result.status]+=1
            self.emit("action_outcome",command_line=request["line"],action=asdict(request["action"]),**asdict(result))
        self.pending_actions=remaining

    def finish(self):
        if self.family:self.checkpoint(self.family);self.family=None
        self.evaluate()
        for request in self.pending_actions:
            self.outcomes["inconclusive"]+=1
            self.emit("action_outcome",command_line=request["line"],action=asdict(request["action"]),
                      status="inconclusive",reason="输入快照结束前没有足够的后置条件证据",receipt_revision=None)
        self.pending_actions.clear()

    def emit(self, event, **values):
        with (self.directory/"timeline.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(dict(event=event,line=self.lines,log_time=self.last_time,**values),ensure_ascii=False)+"\n")
        self.events+=1

    def checkpoint(self, reason):
        o=self.state.observe()
        self.evaluate(o)
        if o.mode!="GT_BATTLEGROUNDS": return
        if o.turn is not None:self.turns.add(o.turn)
        if o.phase!=self.last_phase:
            self.phases[o.phase]+=1;self.last_phase=o.phase
            self.emit("phase",phase=o.phase,turn=o.turn)
        if reason in ("options","choices"):
            d=self.policy.decide(o)
            self.decisions[d.action.kind]+=1
            self.emit("advice",reason=reason,revision=o.revision,turn=o.turn,phase=o.phase,
                      gold=o.gold,tavern_tier=o.tavern_tier,
                      hand=[c.card_id for c in o.hand],board=[c.card_id for c in o.board],
                      shop=[c.card_id for c in o.shop],choices=[c.card_id for c in o.choices],decision=asdict(d))
            if reason=="options":
                self.last_options_actions=o.actions
                self.last_options_cards={opt["id"]:self.state.entities[opt["entity"]].card_id
                                         for opt in self.state.options if opt["entity"] in self.state.entities}
                represented={a.option_id for a in o.actions}
                for opt in self.state.options:
                    entity=self.state.entities.get(opt["entity"])
                    if entity and opt["error"]=="NONE" and opt["id"] not in represented:
                        c=entity.card()
                        known_control=any(x in c.card_id.lower() for x in ("dragbuy","dragsell","reroll_button","techup","lockall"))
                        if c.controller==o.local_player_id and c.card_type not in ("HERO","PLAYER") and not known_control:
                            self.gaps[c.card_id or "unknown_card"]+=1
        gt=self.state.entities.get(self.state.game_entity)
        pt=self.state.entities.get(self.state.players.get(o.local_player_id))
        terminal=(gt and gt.tags.get("STATE") in ("COMPLETE","3")) or (pt and pt.tags.get("PLAYSTATE") in ("WON","LOST","TIED","CONCEDED","4","5","6","8"))
        if terminal and not self.terminal_seen:
            self.terminal_seen=True
            self.terminal_evidence=dict(game_state=gt.tags.get("STATE") if gt else None,
                                        local_playstate=pt.tags.get("PLAYSTATE") if pt else None)
            self.emit("terminal",**self.terminal_evidence)

    def feed(self,line):
        self.lines+=1
        m=LINE.match(line.strip())
        if not m: return
        method,body=m.group("method"),m.group("body").strip()
        stamp=re.match(r"[DWE] ([\d:.]+)",line)
        family="options" if method=="GameState.DebugPrintOptions" else "choices" if method=="GameState.DebugPrintEntityChoices" else None
        if self.family and (family!=self.family or body.startswith("id=")):
            self.checkpoint(self.family)
        if stamp:
            self.first_time=self.first_time or stamp[1];self.last_time=stamp[1]
        if method=="GameState.SendOption":
            sm=re.search(r"selectedOption=(\d+) selectedSubOption=(-?\d+) selectedTarget=(\d+) selectedPosition=(\d+)",body)
            if sm:
                oid,sid,target,pos=map(int,sm.groups())
                obs=self.state.observe()
                candidates=[a for a in self.last_options_actions if a.option_id==oid and a.sub_option_id==(None if sid==-1 else sid)
                            and (a.kind!=Kind.REORDER or a.position==pos)
                            and (a.source==target if a.kind in (Kind.BUY,Kind.SELL) else (a.target or 0)==target)]
                kinds=sorted({a.kind for a in candidates})
                label=kinds[0] if len(kinds)==1 else "unmapped"
                self.human[label]+=1
                action=candidates[0] if len(candidates)==1 else None
                if action and action.kind==Kind.PLAY and pos>0:action=replace(action,position=pos)
                self.emit("client_send_option",turn=obs.turn,kind=label,option_id=oid,sub_option_id=sid,
                          target=target,position=pos,card_id=self.last_options_cards.get(oid),mapping_evidence="latest_completed_options",
                          action=asdict(action) if action else None)
                if action:self.pending_actions.append(dict(action=action,before=obs,line=self.lines))
        if method=="GameState.DebugPrintEntitiesChosen":
            header=re.match(r"id=(\d+) .*EntitiesCount=(\d+)",body)
            if header:
                self.choice_before=self.state.observe();self.choice_id=int(header[1])
            elif selected:=re.match(r"Entities\[\d+\]=(.*)",body):
                self.choice_submitted+=1
                eid=self.state.resolve(selected[1])
                before=self.choice_before
                candidate=next((a for a in before.actions if a.kind==Kind.CHOOSE and a.source==eid and a.choice_id==self.choice_id),None) if before else None
                if candidate:
                    self.choice_mapped+=1
                    self.pending_actions.append(dict(action=candidate,before=before,line=self.lines))
                self.emit("choice_result",choice_id=self.choice_id,source=eid,mapped=candidate is not None)
        if method=="GameState.DebugPrintEntityChoices" and body.startswith("id="):
            ct=re.search(r"ChoiceType=(\w+)",body)
            if ct:self.choices[ct[1]]+=1
        previous_receipt=self.state.resolutions[-1].at_revision if self.state.resolutions else None
        self.state.feed(line)
        if self.state.resolutions and self.state.resolutions[-1].at_revision!=previous_receipt:self.evaluate()
        self.family=family
        if method=="GameState.DebugPrintPower" and (body=="CREATE_GAME" or re.search(r"tag=(BACON_IN_COMBAT_PHASE|PLAYSTATE|STATE|TURN) value=",body)):
            if body=="CREATE_GAME":
                self.terminal_seen=False;self.terminal_evidence=None
                self.last_options_actions=();self.last_options_cards={}
                self.used_receipts.clear();self.choice_before=None;self.choice_id=None
            self.checkpoint("transition")

    def summary(self):
        o=self.state.observe()
        gt=self.state.entities.get(self.state.game_entity)
        pt=self.state.entities.get(self.state.players.get(o.local_player_id))
        return dict(analyzer_version=__version__,mode=o.mode,build=o.build,lines=self.lines,first_log_time=self.first_time,last_log_time=self.last_time,
                    create_game_snapshots=self.state.generation,turns_seen=sorted(self.turns),phase_entries=dict(self.phases),
                    advice_counts=dict(self.decisions),human_send_option_counts=dict(self.human),
                    client_send_option_counts=dict(self.human),choice_results_seen=self.choice_submitted,choice_results_mapped=self.choice_mapped,
                    choice_requests=dict(self.choices),unmapped_friendly_power_occurrences=dict(self.gaps.most_common()),
                    human_choices_submitted=self.choice_submitted,human_choices_mapped=self.choice_mapped,
                    action_outcomes=dict(self.outcomes),pending_outcomes=len(self.pending_actions),
                    server_action_blocks=self.state.stats.get("resolved_action_blocks",0),
                    partial_action_blocks=self.state.stats.get("partial_action_blocks",0),
                    unterminated_blocks=self.state.stats.get("unterminated_blocks",0),
                    terminal_seen=self.terminal_seen,terminal_evidence=self.terminal_evidence,
                    final_terminal_state=dict(game_state=gt.tags.get("STATE") if gt else None,
                                              local_playstate=pt.tags.get("PLAYSTATE") if pt else None),
                    current=dict(phase=o.phase,turn=o.turn,gold=o.gold,tavern_tier=o.tavern_tier,hand=len(o.hand),board=len(o.board),shop=len(o.shop)),
                    notes=["原始日志仅包含客户端实际记录的信息，不包含服务器隐藏信息。",
                           "client_send_option 是客户端提交事件，不表示服务器已接受或 AI 执行；human_* 计数字段仅为旧版本兼容别名。",
                           "choice_result 是日志中的选择结果，不据此区分人工点击与超时自动选择。",
                           "unmapped_friendly_power 是待检查候选，可能含移动或其他非发动操作。",
                           "看到 CREATE_GAME 与终局不证明每个服务器事件都被客户端记录；重连可能产生多个快照。"])


class MatchRecorder:
    def __init__(self, power: Path, directory: Path, player_id=None):
        directory.mkdir(parents=True,exist_ok=False)
        self.directory=directory
        self.power=RawMirror(power,directory/"Power.log")
        scenes=sorted(power.parent.glob("LoadingScreen*.log"),key=lambda p:p.stat().st_mtime_ns)
        self.scene=RawMirror(scenes[-1],directory/"LoadingScreen.log") if scenes else None
        self.analysis=MatchAnalysis(directory,player_id)
        self.pending=b""

    def poll(self):
        chunk=self.power.poll()
        rows=(self.pending+chunk).split(b"\n");self.pending=rows.pop()
        for row in rows:self.analysis.feed(row.decode("utf-8",errors="replace"))
        if self.scene:self.scene.poll()
        return bool(chunk)

    def save(self,status="recording",error=None):
        report=self.analysis.summary()
        report.update(capture_status=status,error=error,unfinished_line_bytes=len(self.pending),
                      raw_files=[self.power.manifest()]+([self.scene.manifest()] if self.scene else []),
                      start_and_terminal_seen=self.analysis.state.generation>0 and self.analysis.terminal_seen)
        write_json(self.directory/"report.json",report)
        return report


def analyze_log(source: Path, directory: Path, player_id=None):
    """Replay a bounded byte snapshot, even if the live source keeps growing."""
    directory.mkdir(parents=True,exist_ok=False)
    analysis=MatchAnalysis(directory,player_id)
    size=source.stat().st_size
    remaining=size;pending=b"";digest=hashlib.sha256()
    with source.open("rb") as f:
        while remaining:
            chunk=f.read(min(65536,remaining))
            if not chunk:raise OSError("日志在分析时被截断，分析不能视为完整快照")
            remaining-=len(chunk);digest.update(chunk)
            rows=(pending+chunk).split(b"\n");pending=rows.pop()
            for row in rows:analysis.feed(row.decode("utf-8",errors="replace"))
    analysis.finish()
    report=analysis.summary()
    report.update(analysis_status="bounded_snapshot",source=str(source),snapshot_bytes=size,
                  snapshot_sha256=digest.hexdigest(),unfinished_line_bytes=len(pending))
    write_json(directory/"report.json",report)
    return report
