"""Stdlib asynchronous reader transport; embedded verbatim in the v2 worker.

Job records survive a Python restart on the same filesystem. Unfinished records
become unknown and are never executed again automatically. No GPU imports.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler


class JobError(Exception):
    def __init__(self, code, reason):
        self.code, self.reason = code, reason
        super().__init__(reason)


class ReaderJobs:
    def __init__(self, root, session, run, *, max_jobs=128, touch=lambda: None):
        self.root = Path(root) / hashlib.sha256(session.encode()).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.session, self.run, self.max_jobs, self.touch = session, run, max_jobs, touch
        self.lock = threading.Lock()
        self.active = False
        self.instance = uuid.uuid4().hex
        for path in self.root.glob('*.json'):
            row = json.loads(path.read_text())
            if row['status'] in {'queued', 'running'}:
                row.update(status='unknown', error='worker_restarted_outcome_unknown')
                self._save(row)

    def _path(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch('[0-9a-f]{64}', job_id):
            raise JobError(400, 'invalid_job_id')
        return self.root / (job_id + '.json')

    def _save(self, row):
        path = self._path(row['job_id'])
        temporary = path.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8') as f:
            json.dump(row, f, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def get(self, job_id):
        with self.lock:
            path = self._path(job_id)
            if not path.exists():
                raise JobError(404, 'unknown_job_id')
            return json.loads(path.read_text())

    def submit(self, job_id, payload, can_start=True):
        if not isinstance(payload, dict):
            raise JobError(400, 'payload_must_be_object')
        # Snapshot input so the caller cannot mutate a queued request.
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        payload = json.loads(encoded)
        with self.lock:
            path = self._path(job_id)
            if path.exists():
                previous = json.loads(path.read_text())
                if previous['request_sha256'] != fingerprint:
                    raise JobError(409, 'job_id_payload_conflict')
                return previous
            if not can_start:
                raise JobError(503, 'not_ready_or_deadline')
            if self.active:
                raise JobError(409, 'another_job_active')
            if len(list(self.root.glob('*.json'))) >= self.max_jobs:
                raise JobError(429, 'job_capacity_no_eviction')
            row = {'job_id':job_id, 'session_id':self.session,
                   'request_sha256':fingerprint, 'status':'queued'}
            self._save(row)
            self.active = True
            self.touch()
            thread = threading.Thread(target=self._execute, args=(row, payload), daemon=True)
            try:
                thread.start()
            except BaseException:
                self.active = False
                row.update(status='unknown', error='thread_start_failed')
                self._save(row)
                raise
            return dict(row)

    def _execute(self, row, payload):
        try:
            with self.lock:
                row['status'] = 'running'
                self._save(row)
            result = self.run(payload)
            with self.lock:
                row.update(status='completed', result=result)
                self._save(row)
        except Exception as exc:
            with self.lock:
                row.pop('result', None)
                row.update(status='failed', error=type(exc).__name__)
                self._save(row)
        finally:
            with self.lock:
                self.active = False
                self.touch()


def reader_job_handler(jobs, token, health, can_start, heartbeat):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, value):
            raw = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Result is already durable; a lost response never cancels work.

        def authorized(self):
            return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer '+token)

        def guarded(self):
            if not self.authorized():
                self.reply(401, {'error':'unauthorized'})
                return False
            if self.headers.get('X-DocQA-Worker-Instance') != jobs.instance:
                self.reply(409, {'error':'worker_instance_changed'})
                return False
            return True

        def do_GET(self):
            if self.path == '/health':
                if not self.authorized():
                    return self.reply(401, {'error':'unauthorized'})
                return self.reply(200, {**health(), 'async_transport':'reader_jobs_v2',
                    'worker_instance':jobs.instance, 'job_active':jobs.active})
            if not self.guarded():
                return
            try:
                if not self.path.startswith('/v1/reader/jobs/'):
                    raise JobError(404, 'unknown_route')
                return self.reply(200, jobs.get(self.path.rsplit('/', 1)[1]))
            except JobError as exc:
                return self.reply(exc.code, {'error':exc.reason})

        def do_POST(self):
            if self.path == '/heartbeat':
                if not self.authorized():
                    return self.reply(401, {'error':'unauthorized'})
                heartbeat()
                return self.reply(200, {'ok':True})
            if not self.guarded():
                return
            try:
                if self.path != '/v1/reader/jobs':
                    raise JobError(404, 'async_reader_route_required')
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                except ValueError:
                    raise JobError(400, 'invalid_content_length')
                if not 0 < length <= 256000:
                    raise JobError(413, 'request_limit')
                data = json.loads(self.rfile.read(length))
                if data.get('session_id') != jobs.session:
                    raise JobError(409, 'session_changed')
                row = jobs.submit(data['job_id'], data['request'], can_start())
                return self.reply(202 if row['status'] in {'queued','running'} else 200, row)
            except JobError as exc:
                return self.reply(exc.code, {'error':exc.reason})
            except (ValueError, KeyError, TypeError, AttributeError):
                return self.reply(400, {'error':'invalid_job_request'})
    return Handler
