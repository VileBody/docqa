from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from docqa_corpus import Corpus

@pytest.fixture(scope='session')
def corpus():return Corpus(Path(__file__).resolve().parents[1])
