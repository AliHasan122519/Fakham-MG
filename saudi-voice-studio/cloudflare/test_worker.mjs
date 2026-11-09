// Offline logic test of the Worker (Node): fake KV + fake upstream fetch.
import worker from "./stable-link-worker.js";
import assert from "node:assert/strict";

const kv = new Map();
const env = { REG_SECRET: "s3cr3t-s3cr3t-s3cr3t", SVS_KV: { put: async (k, v) => kv.set(k, v), get: async (k) => kv.get(k) ?? null } };
const calls = [];
globalThis.fetch = async (u, init) => { calls.push({ u, init }); return new Response(JSON.stringify({ upstream: true }), { status: 200 }); };
const req = (path, init = {}) => worker.fetch(new Request("https://saudi-studio.me.workers.dev" + path, init), env);

let r = await req("/mcp?token=abc", { method: "POST", body: "{}" });
assert.equal(r.status, 503);                                                         // nothing registered yet
r = await req("/_register", { method: "POST", body: JSON.stringify({ secret: "wrong", url: "https://x-y.trycloudflare.com" }) });
assert.equal(r.status, 401);
r = await req("/_register", { method: "POST", body: JSON.stringify({ secret: env.REG_SECRET, url: "https://evil.example.com" }) });
assert.equal(r.status, 400);                                                         // not a quick-tunnel host
r = await req("/_register", { method: "POST", body: JSON.stringify({ secret: env.REG_SECRET, url: "https://good-tunnel-1.trycloudflare.com" }) });
assert.equal(r.status, 200);
r = await req("/mcp?token=abc", { method: "POST", headers: { "content-type": "application/json", host: "ignored" }, body: '{"jsonrpc":"2.0"}' });
assert.equal(r.status, 200);
assert.equal(calls.at(-1).u, "https://good-tunnel-1.trycloudflare.com/mcp?token=abc");   // path + token forwarded to the CURRENT tunnel
// new Colab session → new tunnel URL → same stable link now reaches the new one
r = await req("/_register", { method: "POST", body: JSON.stringify({ secret: env.REG_SECRET, url: "https://good-tunnel-2.trycloudflare.com" }) });
assert.equal(r.status, 200);
await req("/mcp?token=abc", { method: "POST", body: "{}" });
assert.equal(calls.at(-1).u, "https://good-tunnel-2.trycloudflare.com/mcp?token=abc");
// upstream down → 502 JSON, not a crash
globalThis.fetch = async () => { throw new Error("down"); };
r = await req("/mcp?token=abc", { method: "POST", body: "{}" });
assert.equal(r.status, 502);
console.log("worker logic OK");
