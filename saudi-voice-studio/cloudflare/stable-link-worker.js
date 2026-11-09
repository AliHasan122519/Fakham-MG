// Saudi Voice Studio — stable MCP link (optional). Free Cloudflare Workers plan, no card, no domain.
// The Colab notebook registers its current (changing) trycloudflare URL here; connectors always use THIS fixed URL.
// Bindings (Workers dashboard → Settings → Variables/Bindings):
//   KV namespace binding:  SVS_KV
//   Secret (text):         REG_SECRET   — long random string; the same value goes into Colab Secrets as SVS_WORKER_SECRET
const TARGET_RE = /^https:\/\/[a-z0-9-]+\.trycloudflare\.com$/;      // only quick-tunnel hosts: no open proxy / SSRF

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json" } });
}

function safeEqual(a, b) {                                            // constant-time compare
  a = String(a || ""); b = String(b || "");
  if (a.length !== b.length) return false;
  let r = 0;
  for (let i = 0; i < a.length; i++) r |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return r === 0;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/_register") {
      if (request.method !== "POST") return json({ error: "method" }, 405);
      let body = null;
      try { body = await request.json(); } catch (e) { /* fallthrough */ }
      if (!body || !env.REG_SECRET || !safeEqual(body.secret, env.REG_SECRET)) return json({ error: "unauthorized" }, 401);
      if (!TARGET_RE.test(String(body.url || ""))) return json({ error: "bad url" }, 400);
      await env.SVS_KV.put("target", body.url);
      return json({ ok: true });
    }
    if (url.pathname === "/_health") {
      const t = await env.SVS_KV.get("target");
      return json({ registered: !!t });
    }
    const target = await env.SVS_KV.get("target");
    if (!target) return json({ error: "studio offline: run the Colab notebook first" }, 503);
    const dest = new URL(url.pathname + url.search, target);
    const headers = new Headers(request.headers);
    headers.delete("host");
    headers.delete("cf-connecting-ip");
    const init = { method: request.method, headers, redirect: "manual" };
    if (!["GET", "HEAD"].includes(request.method)) init.body = request.body;
    try {
      return await fetch(dest.toString(), init);
    } catch (e) {
      return json({ error: "studio unreachable (the Colab session may have ended)" }, 502);
    }
  },
};
