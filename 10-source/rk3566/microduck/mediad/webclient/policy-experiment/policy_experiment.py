#!/usr/bin/env python3
"""Client for Xduck controlled policy experiments. Python 3.10+, standard library."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


class Client:
    def __init__(self, base_url, timeout=60):
        url = urlsplit(base_url.rstrip('/'))
        if url.scheme != 'http' or not url.hostname or url.path or url.query or url.fragment:
            raise ValueError('Use http://DEVICE:8080')
        self.host, self.port, self.timeout = url.hostname, url.port or 80, timeout

    def request(self, method, path, value=None):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            body = None if value is None else json.dumps(value, allow_nan=False).encode()
            connection.request(method, path, body, {'Content-Type': 'application/json'})
            response = connection.getresponse()
            data = response.read(256 * 1024 + 1)
            if len(data) > 256 * 1024:
                raise RuntimeError('Oversized API response')
            result = json.loads(data)
            if response.status != 200 or 'error' in result:
                raise RuntimeError(result.get('error', result))
            return result
        finally:
            connection.close()

    @staticmethod
    def session(session_id):
        if len(session_id) != 32 or any(c not in '0123456789abcdef' for c in session_id):
            raise ValueError('Invalid session ID')
        return '/api/policy-experiment/sessions/' + session_id

    def configure(self, path):
        with open(path, encoding='utf-8') as stream:
            config = json.load(stream)
        return self.request('POST', '/api/policy-experiment/sessions', config)

    def download(self, session_id, path, delete=True):
        path, partial = Path(path), Path(str(path) + '.part')
        if path.exists() or partial.exists():
            raise FileExistsError('Choose an unused result path')
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            connection.request('GET', self.session(session_id) + '/result')
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(response.read().decode(errors='replace'))
            expected = response.getheader('X-Content-SHA256')
            expected_bytes = int(response.getheader('Content-Length', '-1'))
            digest, received = hashlib.sha256(), 0
            with partial.open('xb') as output:
                while chunk := response.read(65536):
                    output.write(chunk); digest.update(chunk); received += len(chunk)
                output.flush(); os.fsync(output.fileno())
            if received != expected_bytes or digest.hexdigest() != expected:
                raise RuntimeError('Downloaded result failed size/SHA-256 verification')
            os.replace(partial, path)
            if delete:
                self.request('POST', self.session(session_id) + '/acknowledge', {'sha256': expected})
            return {'path': str(path), 'bytes': received, 'sha256': expected, 'device_deleted': delete}
        finally:
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url')
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('capabilities'); commands.add_parser('list')
    configure = commands.add_parser('configure'); configure.add_argument('config')
    for name in ('status', 'initialize', 'start'):
        command = commands.add_parser(name); command.add_argument('id')
    stop = commands.add_parser('stop'); stop.add_argument('id'); stop.add_argument('run_token', nargs='?')
    download = commands.add_parser('download'); download.add_argument('id'); download.add_argument('file'); download.add_argument('--keep', action='store_true')
    delete = commands.add_parser('delete'); delete.add_argument('id'); delete.add_argument('--sha256', default='')
    args = parser.parse_args(); client = Client(args.url)
    if args.action == 'capabilities': result = client.request('GET', '/api/policy-experiment/capabilities')
    elif args.action == 'list': result = client.request('GET', '/api/policy-experiment/sessions')
    elif args.action == 'configure': result = client.configure(args.config)
    elif args.action == 'status': result = client.request('GET', client.session(args.id))
    elif args.action == 'initialize': result = client.request('POST', client.session(args.id) + '/initialize', {})
    elif args.action == 'start': result = client.request('POST', client.session(args.id) + '/start', {})
    elif args.action == 'stop': result = client.request('POST', client.session(args.id) + '/stop', {} if args.run_token is None else {'run_token': args.run_token})
    elif args.action == 'delete': result = client.request('POST', client.session(args.id) + '/acknowledge', {'sha256': args.sha256})
    else: result = client.download(args.id, args.file, not args.keep)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
