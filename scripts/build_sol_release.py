"""Isolated allowlist for the Sol/P1 product; never upload/publish anything."""
import json,hashlib,os,re,shutil,zipfile,argparse
from pathlib import Path
from dotenv import dotenv_values
ROOT=Path(__file__).resolve().parents[1]
def build(dest):
    if dest.exists() or dest.with_suffix('.zip').exists():raise FileExistsError('Use a new version')
    paths=[]
    for folder in ['docqa_rag','docqa_service','docqa_corpus']:
        paths += [p for p in (ROOT/folder).rglob('*.py') if '__pycache__' not in p.parts]
    names=[
        'README.md',
        'LICENSE',
        'pyproject.toml',
        'requirements-stage2-tested.txt',
        'Dockerfile.stage2',
        '.dockerignore',
        '.gitignore',
        '.env.example',
        'profiles/stage_2/txt.json',
        'profiles/stage_2/compose.yaml',
        'profiles/stage_1/prices.json',
        'profiles/submission/luna_p1.json',
        'profiles/submission/sol_p1.json',
        'profiles/submission/sol_p1_v2.json',
        'profiles/submission/sol_p1_runtime.json',
        'profiles/submission/sol_prices.json',
        'profiles/submission/WIRE_RESPONSE_FORMAT.json',
        'profiles/reader_control_v1/P1.json',
        'profiles/reader_control_v1/P1.system.txt',
        'profiles/reader_control_v1/teaching_examples.json',
        'scripts/submission_http.py',
        'scripts/build_sol_release.py',
        'scripts/qwen_worker_v2.py',
        'docs/MODEL_ENDPOINTS.md',
        'docs/FINAL_REPORT.md',
        'docs/REQUIREMENTS_TRACEABILITY.md',
        'docs/REQUIREMENTS_TRACEABILITY.json',
        'docs/API_EXAMPLES.md',
        'docs/PUBLIC_BENCHMARKS.md',
        'docs/submission/READINESS.json',
        'docs/submission/HANDOFF.json',
        'docs/submission/SUBMISSION_PROFILE.json',
        'tests/test_submission_luna.py',
        'tests/test_stronger_submission.py',
        'tests/test_submission_retry.py',
        'tests/test_public_submission.py',
        'tests/conftest.py',
        'examples/submission/HTTP_INPUTS.json',
        'examples/acceptance/alpha.txt',
        'examples/acceptance/source_fidelity_v2/faithful_paraphrase.txt',
        'examples/acceptance/source_fidelity_v2/explicit_equivalence.txt',
        'examples/acceptance/beta.txt',
        'examples/acceptance/hundred.txt',
        'examples/acceptance/source_fidelity_v2/faithful_paraphrase.pack.json',
        'docs/PROJECT_GUIDE.md',
        'docs/RESULTS_GUIDE.md',
        'docs/RUNNING.md',
        'docs/assets/contractnli-historical.png',
        'docs/assets/contractnli-historical.svg',
        'docs/assets/docqa-overview.svg',
        'docs/assets/docqa-pipeline.svg',
        'docs/assets/metrics.json',
        'docs/assets/reader-cost.png',
        'docs/assets/reader-cost.svg',
        'docs/assets/reader-latency.png',
        'docs/assets/reader-latency.svg',
        'scripts/render_readme_visuals.py',
    ]
    paths += [ROOT/n for n in names]
    # Explicit evidence allowlist: no raw traces, evaluator rubrics or obsolete unlabelled logs.
    evidence=['HISTORICAL_INFRASTRUCTURE.json','HISTORICAL_LUNA_EXAMPLES_METADATA.json',
        'SOL_HTTP_SUMMARY.json','SOL_VECTOR_REUSE.json','SOL_READER_COMPARISON.json',
        'SUBMISSION_V2_CHANGES.json','SUBMISSION_V2_VALIDATION.json','SYNTHETIC_HTTP_OUTPUTS.json']
    paths += [ROOT/'docs/submission/evidence'/name for name in evidence]
    tok=ROOT/'artifacts/tokenizers/cdbee75f17c01a7cc42f958dc650907174af0554'
    paths += [tok/n for n in ['provenance.json','tokenizer.json','tokenizer_config.json','LICENSE']]
    secrets=[v for k,v in os.environ.items() if re.search('KEY|TOKEN|SECRET|PASSWORD',k) and len(v)>=12]
    for f in [ROOT/'.env',ROOT.parent/'.env']:
        if f.exists():secrets += [v for k,v in dotenv_values(f).items() if v and len(v)>=12 and re.search('KEY|TOKEN|SECRET|PASSWORD',k)]
    files={}
    for p in sorted(set(paths)):
        rel=str(p.relative_to(ROOT));data=p.read_bytes()
        if p.is_symlink() or any(v.encode() in data for v in secrets) or (str(Path.home())+'/').encode() in data or (b'-----BEGIN '+b'PRIVATE KEY-----') in data:raise ValueError('Unsafe release entry: '+rel)
        files[rel]=hashlib.sha256(data).hexdigest()
    dest.mkdir(parents=True)
    for rel in files:
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target);target.chmod(0o644)
    manifest={'version':'sol-p1-public-allowlist-v2','files':files,'contains_gold':False,'contains_private_traces':False,'contains_weights':False,'contains_evaluator_data':False,'contains_reviewer_rubrics':False,'supported_test_command':'python -m pytest tests','git_history_scanned':False,'scan':'known env/dotenv secret values, user paths, key markers, symlinks; explicit file allowlist'}
    (dest/'RELEASE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=dest.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(dest.rglob('*')):
            if p.is_file():z.write(p,str(Path(dest.name)/p.relative_to(dest)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,h in files.items():assert hashlib.sha256(z.read(dest.name+'/'+rel)).hexdigest()==h
    receipt={'files':len(files),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'hashes_and_crc_verified':True,'known_secrets_scanned':len(secrets)}
    archive.with_suffix('.zip.sha256').write_text(receipt['sha256']+'  '+archive.name+'\n');return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);print(json.dumps(build(p.parse_args().destination),indent=2))
