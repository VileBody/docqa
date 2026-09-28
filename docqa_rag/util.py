from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path

def digest(value) -> str:
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',',':')).encode()
    return hashlib.sha256(data).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data=json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n'
    fd, temp=tempfile.mkstemp(dir=path.parent, prefix='.'+path.name)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)

class UnsupportedPDF(ValueError): pass
class ModelUnavailable(RuntimeError): pass
class ModelOutputInvalid(ValueError):
    """A response was received but violated its contract; never a transient outage."""
    code = 'model_output_invalid'
    retryable = False
class BudgetExceeded(RuntimeError): pass
class InvalidEvidence(ValueError): pass
