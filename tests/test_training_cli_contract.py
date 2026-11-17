"""Freeze the complete pre-refactor CLI contract, including disabled experiments."""
import hashlib
import json

from hsbg.az.train.cli import build_parser


def test_mechanical_extraction_preserves_all_263_argument_contracts():
    additions = {'recruit_start_bank', 'recruit_start_p'}
    payload = [dict(options=a.option_strings, dest=a.dest, default=a.default,
                    type=getattr(a.type, '__name__', None), choices=a.choices,
                    required=a.required, nargs=a.nargs, action=type(a).__name__)
               for a in build_parser()._actions if a.dest not in additions]
    assert len(payload) == 263
    # Captured by executing just the parser declaration from commit 1c579d7.
    assert hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() == (
        '6c74880e3283dbe997d971010ca037b9b23f1221824be06c451dbc6fc35fec43')


def test_curriculum_additions_are_explicit_and_default_off():
    parser=build_parser()
    assert len(parser._actions)==265
    args=parser.parse_args([])
    assert args.recruit_start_bank=='' and args.recruit_start_p==0.
    args=parser.parse_args(['--recruit-start-bank','bank','--recruit-start-p','0.3'])
    assert args.recruit_start_bank=='bank' and args.recruit_start_p==0.3
