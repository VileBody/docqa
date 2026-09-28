"""A fixed common evidence tokenizer; never confused with provider billing tokens."""
from functools import lru_cache
import importlib.metadata
import tiktoken
from .util import digest

@lru_cache(maxsize=1)
def encoder():return tiktoken.get_encoding('cl100k_base')

def count(text):return len(encoder().encode(text,disallowed_special=()))

@lru_cache(maxsize=1)
def binding():
    enc=encoder()
    return {'id':'tiktoken/cl100k_base','version':importlib.metadata.version('tiktoken'),
        'vocabulary_sha256':digest([(k.hex(),v) for k,v in sorted(enc._mergeable_ranks.items())]),
        'special_tokens_sha256':digest(enc._special_tokens),'purpose':'common evidence packing, not native billing'}
