"""Versioned game knowledge; live stats/tags always take precedence.

This is valuation metadata, never permission to create a legal action. Unknown
builds retain live stats but disable stale text, pool and composition knowledge.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re


TRIBES = ('BEAST','DEMON','DRAGON','ELEMENTAL','MECHANICAL','MURLOC','NAGA','PIRATE','QUILBOAR','UNDEAD')


@lru_cache(maxsize=1)
def bundled_catalog():
    return Catalog(Path(__file__).with_name('data')/'heuristic_cards.json')


class Catalog:
    def __init__(self, path):
        raw=Path(path).read_bytes()
        self.digest=hashlib.sha256(raw).hexdigest()
        data=json.loads(raw)
        if data['schema']!=1:raise ValueError('unsupported_strategy_catalog')
        self.cards=data['cards'];self.builds=tuple(data['builds']);self.source=data['source']
        self.by_db={c['dbfId']:key for key,c in self.cards.items() if 'dbfId' in c}

    def compatible(self, build):
        return str(build) in self.builds

    def get(self, card, build):
        return self.cards.get(card.card_id,{}) if self.compatible(build) else {}

    def text(self, card, build):
        text=re.sub('<[^>]*>|\\[x\\]','',self.get(card,build).get('text','')).lower()
        return ' '.join(text.split())

    def tribes(self, card, build):
        meta=self.get(card,build)
        races=set(meta.get('races',[]))
        race=card.tags.get('CARDRACE',meta.get('race',''))
        # Symbolic live races override the static card (including transformations).
        numeric={'20':'BEAST','15':'DEMON','24':'DRAGON','18':'ELEMENTAL','17':'MECHANICAL',
                 '14':'MURLOC','92':'NAGA','23':'PIRATE','43':'QUILBOAR','11':'UNDEAD','26':'ALL'}
        race=numeric.get(str(race),race)
        if (race in TRIBES or race=='ALL') and race not in races:races={race}
        return set(TRIBES) if 'ALL' in races else races & set(TRIBES)

    def mechanic(self, card, build, tag):
        if card.tags.get('SILENCED')=='1':return False
        if tag in card.tags:return str(card.tags[tag]) not in ('0','False','false','')
        return tag in self.get(card,build).get('mechanics',[])

    def normal_id(self, card, build):
        db=self.get(card,build).get('battlegroundsNormalDbfId')
        return self.by_db.get(db,card.card_id)

    def golden(self, card, build):
        return bool(self.get(card,build).get('battlegroundsNormalDbfId')) or card.tags.get('GOLDEN')=='1' or card.tags.get('PREMIUM')=='1'
