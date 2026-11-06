"""Read-only interruption checks on supplied frames; never dismiss a popup."""
import time
import numpy as np
from .contracts import Blocked
from .vision import OCR,crop_box


class InterruptionGuard:
    labels=frozenset(('重新连接','连接已断开','连接失败','断开连接','网络错误','返回游戏','认输','选项','验证码','登录','发生错误'))
    def __init__(self,ocr=None,*,clock=time.monotonic):
        self.ocr=ocr;self.clock=clock;self.key=None;self.baseline=None;self.blocked=None;self.next_probe=0.;self.scans=0
        self.choice_baseline=None;self.last_probe=None
    def check(self,image,data,pending=None,*,choice_regions=(),choice_visible=False):
        from .lifecycle import physical_context
        interaction=data.get('interaction') or {}
        key=physical_context(data,pending)
        levels=np.asarray([np.asarray(image.crop(crop_box(box,image.size)).convert('RGB'),dtype=np.float32).mean()
            for box in ([.04,.03,.15,.06],[.86,.45,.08,.15],[.12,.70,.06,.10])])
        same=key==self.key
        dim=bool(same and self.baseline is not None and np.sum((self.baseline>15)&(levels<self.baseline*.65))>=2)
        cards=None
        owned_choice=bool(choice_regions and data.get('choices') and data.get('ready') and interaction.get('mode')=='choice'
            and interaction.get('owner')==data.get('local_player_id') and data.get('local_player_id') is not None
            and interaction.get('display_state') not in ('hidden','waiting') and len(choice_regions)==len(data['choices']))
        if owned_choice:
            cards=np.asarray([np.asarray(image.crop(crop_box(box,image.size)).convert('RGB'),dtype=np.float32).mean()
                              for box in choice_regions])
        expected_backdrop=bool(dim and cards is not None and choice_visible
            and len(cards)==len(data['choices']) and np.all(cards>30)
            and (self.choice_baseline is None or np.all(cards>=self.choice_baseline*.5)))
        now=self.clock()
        if same and not self.blocked and not dim:return
        if same and self.blocked and now<self.next_probe:raise Blocked(self.blocked)
        self.ocr=self.ocr or OCR();self.scans+=1
        # Keep the entire screen: no popup/control area is omitted. Half-size
        # OCR was verified on saved client frames; physical coordinates and
        # admission frames always remain at native resolution.
        sample=image
        if image.width>1280:
            from PIL import Image
            sample=image.resize((1280,round(image.height*1280/image.width)),Image.Resampling.LANCZOS)
        scan_started=self.clock();rows=self.ocr.read(sample)
        phrases={label for label in self.labels if len(label)>=3}
        found=[text.strip() for text,confidence in rows if confidence>=.8 and
               (text.strip() in self.labels or any(label in ''.join(text.split()) for label in phrases))]
        self.last_probe=dict(seconds=self.clock()-scan_started,width=sample.width,expected_choice_backdrop=expected_backdrop,
                             choice_visible=bool(choice_visible),
                             background_levels=levels.tolist(),choice_levels=cards.tolist() if cards is not None else None)
        self.key=key;self.next_probe=self.clock()+.5
        self.blocked=('client interruption: '+','.join(found)) if found else 'unexpected dimmed overlay; input paused' if dim and not expected_backdrop else None
        if self.blocked:raise Blocked(self.blocked)
        self.baseline=levels
        self.choice_baseline=cards
