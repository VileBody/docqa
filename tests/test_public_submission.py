"""Self-contained public-release consistency checks; no private fixtures or models."""
from pathlib import Path
from docqa_service.config import ROOT
from docqa_rag.util import read,digest


def test_public_examples_use_only_exact_distributable_txt_spans():
    data=read(ROOT/'docs/submission/evidence/SYNTHETIC_HTTP_OUTPUTS.json')
    assert data['execution']['model']=='gpt-6-sol'
    assert data['execution']['profile']=='submission-sol-p1-v1' and data['execution']['max_attempts']==1
    assert data['current_submission']=='submission-sol-p1-v2' and data['current_max_attempts']==3
    assert data['new_live_calls']==0 and len(data['events'])==9
    ids=set()
    for row in data['events']:
        ids.add(row['case_id']);relative=row['source']['document_path']
        assert relative.startswith('examples/acceptance/')
        path=(ROOT/relative).resolve();assert path.is_relative_to(ROOT.resolve())
        text=path.read_bytes().decode('utf8')
        assert digest(path.read_bytes())==row['source']['document_sha256']
        assert set(row['response'])=={'answer','found','status','support_check','citations'}
        for citation in row['response']['citations']:
            assert text[citation['char_start']:citation['char_end']]==citation['text']
            assert text[:citation['char_start']].count('\f')+1==citation['page']
            assert citation['document_id']==row['request']['document_id']
    assert ids=={'H01','H03','H04','H05','H10','H11','H12','faithful_paraphrase','explicit_equivalence'}
    h03=next(x for x in data['events'] if x['case_id']=='H03')['response']
    assert h03['status']=='not_found' and h03['found'] is False and h03['citations']==[]


def test_current_docs_reference_correct_profile_and_distinct_evidence():
    hand=read(ROOT/'docs/submission/HANDOFF.json')
    assert hand['active_profile']=='profiles/submission/sol_p1_v2.json'
    assert hand['profile_sha256']==digest((ROOT/hand['active_profile']).read_bytes())
    rows={x['id']:x for x in read(ROOT/'docs/REQUIREMENTS_TRACEABILITY.json')}
    assert 'false found/complete. Other 11' not in rows['T07']['verification']
    assert '3 attempts total' in rows['T25']['verification']
    assert '12 actual LangChain reader' in rows['T22']['verification']
    assert 'Sol v1 14' in rows['T27']['verification']
    for row in rows.values():
        for ref in row['evidence']:assert (ROOT/ref).is_file(),ref
    for name in ['README.md','docs/API_EXAMPLES.md','docs/REQUIREMENTS_TRACEABILITY.md']:
        assert 'SYNTHETIC_HTTP_OUTPUTS.json' in (ROOT/name).read_text()


def test_manifest_when_running_from_a_public_release():
    path=ROOT/'RELEASE_MANIFEST.json'
    if not path.exists():return # Build-independent workspace check; packaging verifies this again in release.
    manifest=read(path)
    assert not manifest['contains_gold'] and not manifest['contains_private_traces']
    for name,expected in manifest['files'].items():
        assert not set(Path(name).parts)&{'runs','evaluator','gold','.git','.operator'}
        assert Path(name).name not in {'.env','.worker.env','TASK.md','SYNTHETIC_HTTP_OUTPUTS_LUNA_RAW.json'}
        assert digest((ROOT/name).read_bytes())==expected
    assert (ROOT/'artifacts/tokenizers/cdbee75f17c01a7cc42f958dc650907174af0554/LICENSE').is_file()
