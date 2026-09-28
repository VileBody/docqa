"""Bounded retries for read-only retrieval; ambiguous writes are not repeated."""
import time
import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse


def read_with_retry(call):
    for attempt in range(3):
        try:
            return call()
        except (ResponseHandlingException, UnexpectedResponse, httpx.TransportError) as exc:
            source = getattr(exc, 'source', exc)
            transient = isinstance(source,httpx.TransportError) or getattr(exc,'status_code',None) in {408,429,500,502,503,504}
            if not transient or attempt == 2:
                raise
            time.sleep(.2 * (2 ** attempt))


class ReadRetryClient(QdrantClient):
    def query_points(self,*args,**kwargs):
        parent=super().query_points
        return read_with_retry(lambda:parent(*args,**kwargs))

    def get_collection(self,*args,**kwargs):
        parent=super().get_collection
        return read_with_retry(lambda:parent(*args,**kwargs))

    def scroll(self,*args,**kwargs):
        parent=super().scroll
        return read_with_retry(lambda:parent(*args,**kwargs))
