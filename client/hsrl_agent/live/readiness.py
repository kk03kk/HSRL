"""Offline readiness is separate from a successful model/encoder unit test."""
from hsbg.env import obs_std as O

# These are internal simulator representations, not aliases of arbitrary client
# tag counts. Completing them needs a reviewed semantic adapter, not imputation.
UNRESOLVED={
    'effects_len':'length of simulator Player.effects; not number of client enchantments',
    'hand_queue_len':'simulator deferred hand queue; not visible hand size',
    'max_gold_bonus':'future gold-cap bonus; MAXRESOURCES=99 is not this quantity',
    'tavern_atk':'simulator persistent tavern attack state needs effect-specific interpretation',
    'tavern_hp':'simulator persistent tavern health state needs effect-specific interpretation',
    'chronum':'saved versus currently spendable Chronum requires visit lifecycle',
}


def report(profile=None,coverage=None,archive=None):
    profile=profile or {};coverage=coverage or {};archive=archive or {}
    missing=coverage.get('missing_fields',{}) or {k:1 for k in coverage.get('unknown',[])}
    contract=[dict(field=name,scale=O.GLOBAL_FIELD_SCALE.get(name,1.),
        semantics=UNRESOLVED.get(name,'explicit scoped tags, complete event history or derived public state required'),
        unresolved_internal_representation=name in UNRESOLVED) for name in O.GLOBAL_FIELD_NAMES]
    return dict(format='hsrl.live-readiness.v1',model_contract='3227 floats / 124 cards / 245 actions',
        input_ready=False,ready_for='offline rehearsal and human-started read-only calibration/shadow',
        unresolved_global_semantics=UNRESOLVED,global_contract=contract,observed_missing=missing,
        historical_images=archive.get('images'),historical_builds=archive.get('builds',{}),
        historical_coordinate_transforms=archive.get('counts',{}).get('transform_verified',0),
        profile_review_state=profile.get('review_state'),
        selection_classifier_groups=list(profile.get('selection_guards',{})),
        branch_classifier_sources=list(profile.get('branch_guards',{})),
        magnetic_geometry_qualified=bool(profile.get('magnetize_drop',{}).get('qualified')),
        remaining=[
            'complete and validate real public-to-model observation semantics, including seen-board timing',
            'qualify selection/hover/transition and source-specific branch perception on independent real captures',
            'check current client executable/build/DPI/window against inherited coordinates at first human-started test',
            'validate input hit timing, causal receipts and magnetic insertion on the actual client'],
        game_started=False,native_backend_created=False,
        interpretation='This report does not authorize input; decoder tests and historical overlays do not prove live readiness.')
