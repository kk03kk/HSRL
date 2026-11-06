"""Read-only monitoring of a live run; no input endpoint."""
import json,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

PAGE='''<!doctype html><meta charset="utf-8"><title>Game Bridge · 实机运行</title>
<style>body{font:17px system-ui;background:#122036;color:#edf5ff;margin:20px}pre{white-space:pre-wrap}img{max-width:100%}.cols{display:grid;grid-template-columns:2fr 1fr;gap:20px}</style>
<h2>真实客户端输入运行 · F8 紧急停止</h2><p id="status">连接中</p>
<div class="cols"><img id="frame"><pre id="events"></pre></div>
<script>async function update(){try{let s=await(await fetch('/state')).json();document.getElementById('status').textContent=`阶段 ${s.observation?.phase||'?'} · 回合 ${s.observation?.turn??'?'} · 金币 ${s.observation?.gold??'?'} · ${s.report?.status||'running'} ${s.report?.reason||''}`;document.getElementById('events').textContent=JSON.stringify({action:s.action,verification:s.verification,report:s.report,window_blockers:s.blockers},null,2);if(s.frame)document.getElementById('frame').src='/frame?t='+s.frame;}catch(e){document.getElementById('status').textContent=e.message}}update();setInterval(update,700);</script>'''

def start_panel(provider,port=8796):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.path=='/':data=PAGE.encode();mime='text/html; charset=utf-8'
            elif self.path=='/state':
                rows=[];path=provider.directory/'agent.jsonl'
                if path.exists():
                    with path.open('rb') as f:
                        f.seek(max(0,path.stat().st_size-65536));raw=f.read()
                    for line in raw.splitlines():
                        try:rows.append(json.loads(line))
                        except ValueError:pass
                def latest(event):return next((r for r in reversed(rows) if r.get('event')==event),None)
                final=provider.directory/'report.json'
                result=json.loads(final.read_text(encoding='utf-8')) if final.exists() else None
                data=json.dumps(dict(observation=provider.last_obs.to_dict() if provider.last_obs else None,
                    action=latest('decision'),verification=latest('verification'),report=result,
                    blockers=provider.last_row.get('blockers',[]) if provider.last_row else provider.last_window.blockers,
                    frame=provider.last_row['id'] if provider.last_row else None),ensure_ascii=False).encode();mime='application/json'
            elif self.path.startswith('/frame'):
                image=provider.last_image
                if image is None:self.send_error(404);return
                import io
                buffer=io.BytesIO();image.copy().save(buffer,format='JPEG',quality=80);data=buffer.getvalue();mime='image/jpeg'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True
    threading.Thread(target=server.serve_forever,daemon=True).start();return server
