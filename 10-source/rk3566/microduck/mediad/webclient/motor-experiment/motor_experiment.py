#!/usr/bin/env python3
"""Uploaded motor tasks v4. Python 3.10+, standard library; no real-time control API."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while chunk := stream.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def write_task(path, header, frames):
    """Stream a complete 20 ms NDJSON timeline; publish only after validation."""
    duration = header.get('duration_ms')
    if header.get('period_ms') != 20 or type(duration) is not int or not 0 < duration <= 1800000 or duration % 20:
        raise ValueError('Require 20 ms periods and duration 20..1800000 ms divisible by 20')
    path = Path(path)
    partial = path.with_name(path.name+'.part')
    if path.exists() or partial.exists():
        raise FileExistsError('Choose an unused task file path')
    count, size = 0, 0
    try:
        with partial.open('x', encoding='utf-8') as output:
            first = json.dumps(header, allow_nan=False, separators=(',', ':'))+'\n'
            if len(first.encode()) > 4096:
                raise ValueError('Header exceeds 4096 bytes')
            output.write(first); size += len(first.encode())
            for index, frame in enumerate(frames):
                if index >= duration//20:
                    raise ValueError('Too many frames for declared duration')
                if type(frame.get('at_ms')) is not int or frame['at_ms'] != index*20:
                    raise ValueError(f'Frame {index}: expected at_ms={index*20}')
                row = json.dumps(frame, allow_nan=False, separators=(',', ':'))+'\n'
                size += len(row.encode())
                if len(row.encode()) > 4096 or size > 128*1024*1024:
                    raise ValueError('Task or frame exceeds size limit')
                output.write(row); count += 1
            if count*20 != duration:
                raise ValueError('Frame count does not match duration')
            output.flush(); os.fsync(output.fileno())
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return dict(bytes=path.stat().st_size, sha256=sha256_file(path), frames=count)


class Client:
    def __init__(self, base_url, *, timeout=60):
        url = urlsplit(base_url.rstrip('/'))
        if url.scheme != 'http' or not url.hostname or url.username or url.query or url.fragment or url.path:
            raise ValueError('Use http://DEVICE:8080')
        self.host, self.port, self.timeout = url.hostname, url.port or 80, timeout

    def _connection(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    @staticmethod
    def _id(task_id):
        if len(task_id) != 32 or any(c not in '0123456789abcdef' for c in task_id):
            raise ValueError('Invalid task ID')
        return '/api/motor-experiment/tasks/'+task_id

    @staticmethod
    def _result(response):
        data = response.read(256*1024+1)
        if len(data) > 256*1024:
            raise RuntimeError('Oversized API response')
        value = json.loads(data)
        if response.status != 200 or 'error' in value:
            raise RuntimeError(value.get('error', value))
        return value

    def _json(self, method, path, value=None):
        connection = self._connection()
        try:
            body = None if value is None else json.dumps(value, allow_nan=False).encode()
            connection.request(method, path, body, {'Content-Type': 'application/json'})
            return self._result(connection.getresponse())
        finally:
            connection.close()

    def capabilities(self):
        return self._json('GET', '/api/motor-experiment/capabilities')

    def tasks(self):
        return self._json('GET', '/api/motor-experiment/tasks')

    def upload(self, path):
        path = Path(path)
        size, digest = path.stat().st_size, sha256_file(path)
        connection = self._connection()
        try:
            with path.open('rb') as stream:
                connection.request('POST', '/api/motor-experiment/tasks/upload', stream,
                                   {'Content-Length': str(size), 'X-Content-SHA256': digest,
                                    'Content-Type': 'application/x-ndjson'})
                result = self._result(connection.getresponse())
            if result.get('input_sha256') != digest or result.get('state') != 'ready':
                raise RuntimeError('Upload was not validated as ready')
            return result
        finally:
            connection.close()

    def start(self, task_id):
        """Explicit actuator operation. Uploading alone never starts a task."""
        return self._json('POST', self._id(task_id)+'/start', {})

    def stop(self, task_id, run_token=None):
        """Stop only the run started by the caller that received this token."""
        return self._json('POST', self._id(task_id)+'/stop',
                          {} if run_token is None else {'run_token': run_token})

    def status(self, task_id):
        return self._json('GET', self._id(task_id))

    def acknowledge(self, task_id, local_path):
        """Only acknowledge a complete local file whose hash matches this task."""
        digest = sha256_file(local_path)
        status = self.status(task_id)
        if digest != status.get('result_sha256') or Path(local_path).stat().st_size != status.get('result_bytes'):
            raise RuntimeError('Local result does not match task; device data retained')
        return self._json('POST', self._id(task_id)+'/acknowledge', {'sha256': digest})

    def download(self, task_id, path, *, delete=True):
        path = Path(path)
        partial = path.with_name(path.name+'.part')
        if path.exists() or partial.exists():
            raise FileExistsError('Choose an unused result path; existing files are preserved')
        connection = self._connection()
        try:
            connection.request('GET', self._id(task_id)+'/result')
            response = connection.getresponse()
            if response.status != 200:
                self._result(response)
            expected = response.getheader('X-Content-SHA256')
            size = int(response.getheader('Content-Length', '-1'))
            received, digest = 0, hashlib.sha256()
            with partial.open('xb') as output:
                while chunk := response.read(65536):
                    output.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if received != size or digest.hexdigest() != expected:
                raise RuntimeError('Result size/hash mismatch; device data retained')
            # Recheck against status, then publish the local file before deleting remotely.
            status = self.status(task_id)
            if expected != status.get('result_sha256') or received != status.get('result_bytes'):
                raise RuntimeError('Result differs from finalized task; device data retained')
            os.replace(partial, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            if delete:
                self.acknowledge(task_id, path)
            return dict(path=str(path), bytes=received, sha256=expected, device_deleted=delete)
        finally:
            connection.close()

    def download_logs(self, path):
        connection = self._connection()
        try:
            connection.request('GET', '/api/logs?format=jsonl')
            response = connection.getresponse()
            if response.status != 200:
                self._result(response)
            with open(path, 'xb') as output:
                while chunk := response.read(65536):
                    output.write(chunk)
        finally:
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url', help='http://DEVICE:8080')
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('capabilities')
    sub.add_parser('list')
    upload = sub.add_parser('upload'); upload.add_argument('file')
    for action in ['status', 'start']:
        command = sub.add_parser(action); command.add_argument('id')
    stop = sub.add_parser('stop'); stop.add_argument('id'); stop.add_argument('run_token', nargs='?')
    download = sub.add_parser('download'); download.add_argument('id'); download.add_argument('file')
    download.add_argument('--keep', action='store_true', help='Keep device copy after verified download')
    ack = sub.add_parser('acknowledge'); ack.add_argument('id'); ack.add_argument('file')
    logs = sub.add_parser('logs'); logs.add_argument('file')
    args = parser.parse_args()
    client = Client(args.url)
    if args.action == 'capabilities': result = client.capabilities()
    elif args.action == 'list': result = client.tasks()
    elif args.action == 'upload': result = client.upload(args.file)
    elif args.action == 'download': result = client.download(args.id, args.file, delete=not args.keep)
    elif args.action == 'acknowledge': result = client.acknowledge(args.id, args.file)
    elif args.action == 'logs': result = client.download_logs(args.file)
    elif args.action == 'stop': result = client.stop(args.id, args.run_token)
    else: result = getattr(client, args.action)(args.id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
