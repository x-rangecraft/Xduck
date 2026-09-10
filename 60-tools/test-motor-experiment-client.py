#!/usr/bin/env python3
"""Task client regression with a local HTTP double; never connects to real motors."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'10-source/rk3566/microduck/mediad/webclient'))
from motor_experiment import Client, write_task
TASK='a'*32
DATA=b'finalized task archive bytes'
HASH=hashlib.sha256(DATA).hexdigest()

class Handler(BaseHTTPRequestHandler):
    calls=[]
    corrupt=False
    def log_message(self,*args):pass
    def reply(self,value):
        data=json.dumps(value).encode();self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        self.calls.append(('GET',self.path,None))
        if self.path.endswith('/result'):
            data=b'corrupted'+DATA if self.corrupt else DATA
            self.send_response(200);self.send_header('Content-Length',str(len(data)));self.send_header('X-Content-SHA256',HASH);self.end_headers();self.wfile.write(data)
        elif self.path.endswith('/capabilities'):self.reply({'version':3,'ready':True})
        else:self.reply({'id':TASK,'state':'completed','result_sha256':HASH,'result_bytes':len(DATA)})
    def do_POST(self):
        data=self.rfile.read(int(self.headers.get('Content-Length','0')))
        self.calls.append(('POST',self.path,data))
        if self.path.endswith('/upload'):
            assert self.headers['X-Content-SHA256']==hashlib.sha256(data).hexdigest()
            self.reply({'id':TASK,'state':'ready','input_sha256':self.headers['X-Content-SHA256']})
        elif self.path.endswith('/start'):self.reply({'id':TASK,'state':'preparing','run_token':'b'*32})
        else:self.reply({'ok':True})

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler);cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start();cls.url=f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def setUp(self):Handler.calls.clear();Handler.corrupt=False
    def test_upload_never_starts_and_streams_exact_file(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'task.jsonl';p.write_bytes(b'{}\n');result=Client(self.url).upload(p)
            self.assertEqual(result['state'],'ready');self.assertEqual([x[1] for x in Handler.calls],['/api/motor-experiment/tasks/upload'])
    def test_stop_carries_the_run_token_returned_by_start(self):
        client=Client(self.url);started=client.start(TASK);client.stop(TASK,started['run_token'])
        stop=[c for c in Handler.calls if c[1].endswith('/stop')][0]
        self.assertEqual(json.loads(stop[2]),{'run_token':'b'*32})
    def test_download_verifies_before_deletion_and_preserves_local_file(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.tar';result=Client(self.url).download(TASK,p)
            self.assertEqual(p.read_bytes(),DATA);self.assertTrue(result['device_deleted'])
            acks=[c for c in Handler.calls if c[1].endswith('/acknowledge')]
            self.assertEqual(len(acks),1);self.assertEqual(json.loads(acks[0][2]),{'sha256':HASH})
    def test_corrupt_download_never_deletes_device_copy(self):
        Handler.corrupt=True
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.tar'
            with self.assertRaisesRegex(RuntimeError,'mismatch'):Client(self.url).download(TASK,p)
            self.assertFalse(p.exists());self.assertFalse(any(c[1].endswith('/acknowledge') for c in Handler.calls))
    def test_keep_and_wrong_local_ack_do_not_delete(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.tar';Client(self.url).download(TASK,p,delete=False);p.write_bytes(b'bad')
            with self.assertRaisesRegex(RuntimeError,'does not match'):Client(self.url).acknowledge(TASK,p)
            self.assertFalse(any(c[1].endswith('/acknowledge') for c in Handler.calls))
    def test_legacy_live_methods_are_removed(self):
        client=Client(self.url)
        for name in ['acquire','command','release','feedback','probe']:self.assertFalse(hasattr(client,name))
    def test_generator_requires_complete_exact_timeline(self):
        with tempfile.TemporaryDirectory() as d:
            h={'period_ms':20,'duration_ms':40}
            with self.assertRaisesRegex(ValueError,'Frame 1'):write_task(Path(d)/'bad.jsonl',h,[{'at_ms':0},{'at_ms':30}])
            write_task(Path(d)/'ok.jsonl',h,[{'at_ms':0},{'at_ms':20}])

if __name__=='__main__':unittest.main()
