"""Offline document-QA corpus, benchmark views, and deterministic mutations."""
from .core import Corpus, Document
from .benchmark import Benchmark
from .sensitivity import SensitivitySuite
__version__="0.2.1"
__all__=["Corpus","Document","Benchmark","SensitivitySuite","MutationGenerator","PublicAcquisition"]
def __getattr__(name: str):
    if name=="PublicAcquisition":
        from .acquisition import PublicAcquisition
        return PublicAcquisition
    if name=="MutationGenerator":
        from .mutations import MutationGenerator
        return MutationGenerator
    raise AttributeError(name)
