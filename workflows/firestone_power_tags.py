"""Spendable gold from the selected player's Power.log tags."""
RESOURCE_TAGS = ("26", "25", "295")


def available_gold(tags):
    base,used,temporary = RESOURCE_TAGS
    if base not in tags:
        return None
    try:
        return max(0,int(tags[base])+int(tags.get(temporary,0))-int(tags.get(used,0)))
    except (TypeError,ValueError):
        return None


def remaining_health(tags):
    if "45" not in tags:
        return None
    try:
        return max(0,int(tags["45"])-int(tags.get("44",0)))
    except (TypeError,ValueError):
        return None
