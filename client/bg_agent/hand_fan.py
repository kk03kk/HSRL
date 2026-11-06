"""Experimental fan correspondence, deliberately NOT an input-region mapper.

Tier stars identify a sequence of visible badges, not card art. A unique fit
does not establish a draggable portrait area or current interactability.
"""
from itertools import combinations
import cv2
import numpy as np


def tier_markers(image, badges):
    width, height = image.size
    markers = []
    for box in badges:
        if not (box.y > height*.82 and width*.22 < box.x < width*.8
                and box.width < width*.026 and height*.017 < box.height < height*.04):
            continue
        pad = round(min(box.width, box.height)*.18)
        rgb = np.asarray(image.crop((box.x-pad, box.y-pad,
                                     box.x+box.width+pad, box.y+box.height+pad)))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        mask = ((h > 15) & (h < 45) & (s > 90) & (v > 160)).astype('uint8')
        _, _, stats, centers = cv2.connectedComponentsWithStats(mask)
        stars = [int(area) for (_, _, _, _, area), (x, y) in zip(stats[1:], centers[1:])
                 if box.width*box.height*.012 < area < box.width*box.height*.2
                 and pad*.3 < x < rgb.shape[1]-pad*.3
                 and pad*.3 < y < rgb.shape[0]-pad*.3]
        if 1 <= len(stars) <= 7:
            markers.append(dict(rect=vars(box), tier=len(stars),
                                x=box.x+box.width/2, y=box.y+box.height/2))
    return sorted(markers, key=lambda m:m['x'])


def match_fan(cards, markers, image_size):
    """Require all minion tiers, unique monotone assignment and measured fan.

    Missing spell badges are expected. Missing minion badges are not filled
    by interpolation. Extra artifacts may remain unmatched, but alternative
    assignments cause a refusal. Results are diagnostic evidence only.
    """
    width, height = image_size
    result = dict(status='unknown', reason='insufficient_supported_minions',
                  assignments=[], draggable_region='unknown', executable=False)
    if not 3 <= len(cards) <= 10:
        return result
    if sorted(c.position for c in cards) != list(range(1, len(cards)+1)):
        return dict(result, reason='log_slot_order_unknown')
    minions = sorted((c for c in cards if c.card_type == 'MINION'), key=lambda c:c.position)
    if len(minions) < 4 or any(c.tier not in range(1, 8) for c in minions):
        return result
    if len(markers) > 24:
        return dict(result, reason='excessive_visual_candidates')
    fits = {}
    positions = np.array([c.position-1 for c in minions], dtype=float)
    # Two independently observed endpoints propose scale/origin. Every other
    # logged minion must independently match a badge with the right star count.
    first, last = minions[0], minions[-1]
    for a, b in combinations(range(len(markers)), 2):
        if markers[a]['tier'] != first.tier or markers[b]['tier'] != last.tier:
            continue
        pitch = (markers[b]['x']-markers[a]['x'])/(last.position-first.position)
        if not width*.018 < pitch < width*.075:
            continue
        origin = markers[a]['x']-(first.position-1)*pitch
        chosen = []
        for card in minions:
            candidates = [i for i, marker in enumerate(markers)
                          if marker['tier'] == card.tier
                          and abs(marker['x']-(origin+(card.position-1)*pitch)) < pitch*.14]
            if len(candidates) != 1:
                break
            chosen.append(candidates[0])
        if len(chosen) != len(minions) or chosen != sorted(set(chosen)):
            continue
        xs = np.array([markers[i]['x'] for i in chosen])
        ys = np.array([markers[i]['y'] for i in chosen])
        fitted_pitch, fitted_origin = np.polyfit(positions, xs, 1)
        error = float(np.max(np.abs(xs-(fitted_origin+positions*fitted_pitch))))
        # Quadratic bow must be convex and explain observed badge heights.
        centered = (xs-float(np.mean(xs)))/width
        curve = np.polyfit(centered, ys/height, 2)
        y_error = float(np.max(np.abs(np.polyval(curve, centered)-ys/height)))
        if error > fitted_pitch*.1 or not .02 < curve[0] < 8 or y_error > .009:
            continue
        fits[tuple(chosen)] = dict(pitch=float(fitted_pitch), x_residual=error,
                                  y_residual_fraction=y_error, bow=float(curve[0]))
    if len(fits) != 1:
        return dict(result, reason='ambiguous_pattern' if fits else 'tier_or_fan_geometry_mismatch',
                    candidate_assignments=len(fits))
    chosen, evidence = next(iter(fits.items()))
    return dict(result, status='inferred', reason='unique_tier_sequence_and_fan_geometry',
                assignments=[dict(entity_id=c.entity_id, position=c.position,
                                  tier=c.tier, badge=markers[i]['rect'])
                             for c, i in zip(minions, chosen)], evidence=evidence)


def diagnose_fan(cards, image, badges):
    markers = tier_markers(image, badges)
    return dict(match_fan(cards, markers, image.size), markers=markers)


def main():
    import argparse
    import hashlib
    import json
    from collections import Counter
    from pathlib import Path
    from PIL import Image, ImageDraw
    from .model import Observation
    from .spatial import SpatialMapper
    parser=argparse.ArgumentParser(description='Offline non-executable fan correspondence diagnostic')
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    from .resources import calibration_root
    parser.add_argument('--profile',type=Path,default=calibration_root()/'v05_ui/profile.json')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    mapper=SpatialMapper(args.profile)
    results=[]
    for path in sorted(args.source.glob('frame-*.json')):
        if not path.stem.removeprefix('frame-').isdigit():continue
        row=json.loads(path.read_text(encoding='utf-8'))
        if 'observation' not in row or 'capture' not in row:continue
        obs=Observation.from_dict(row['observation'])
        with Image.open(row['capture']['path']) as opened:image=opened.convert('RGB')
        _,badges=mapper.components(image)
        result=diagnose_fan(obs.hand,image,badges)
        result.update(id=row['id'],hand_count=len(obs.hand),phase=obs.phase,
                      capture_sha256=hashlib.sha256(Path(row['capture']['path']).read_bytes()).hexdigest(),
                      metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      blockers=row.get('blockers',[]), review='unreviewed',
                      contribution_to_mapping_coverage=0)
        if result['status']=='inferred':
            draw=ImageDraw.Draw(image)
            for assignment in result['assignments']:
                box=assignment['badge'];x,y,w,h=[box[k] for k in ('x','y','width','height')]
                draw.rectangle((x,y,x+w,y+h),outline='yellow',width=3)
                draw.text((x,y-16),f"slot {assignment['position']}: tier {assignment['tier']}",fill='yellow')
            draw.text((20,20),'DIAGNOSTIC BADGES ONLY - NO DRAG REGION / NO INPUT',fill='yellow')
            image.save(args.out/(row['id']+'-badges.png'))
        results.append(result)
    report=dict(schema=1,source=str(args.source.resolve()),frames=len(results),
                status_counts=dict(Counter(r['status'] for r in results)),
                reason_counts=dict(Counter(r['reason'] for r in results)),
                inferred_badges=sum(len(r['assignments']) for r in results),
                mapping_coverage_increment=0,
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                samples=results)
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},ensure_ascii=False))


if __name__=='__main__':main()
