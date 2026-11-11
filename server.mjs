// خادم ثابت بسيط بدون اعتمادات إضافية
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
const root = process.cwd();
const types = { '.html':'text/html; charset=utf-8', '.js':'text/javascript', '.mjs':'text/javascript', '.css':'text/css', '.json':'application/json', '.wav':'audio/wav', '.png':'image/png', '.svg':'image/svg+xml', '.woff2':'font/woff2' };
export function start(port = 4173) {
  return http.createServer((req, res) => {
    let p = path.join(root, decodeURIComponent(req.url.split('?')[0]));
    if (!p.startsWith(root)) { res.writeHead(403).end(); return; }
    if (fs.existsSync(p) && fs.statSync(p).isDirectory()) p = path.join(p, 'index.html');
    fs.readFile(p, (e, d) => {
      if (e) { res.writeHead(404).end('not found'); return; }
      res.writeHead(200, { 'Content-Type': types[path.extname(p)] || 'application/octet-stream' }).end(d);
    });
  }).listen(port, () => console.log(`http://localhost:${port}`));
}
if (import.meta.url === `file://${process.argv[1]}`) start(Number(process.env.PORT) || 4173);
