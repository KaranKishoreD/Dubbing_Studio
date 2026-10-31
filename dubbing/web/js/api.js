// Calls to the studio's JSON API (server/app.py).

async function request(method, url, body) {
  const res = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
  return data;
}

const get = (url) => request("GET", url);
const post = (url, body) => request("POST", url, body);
const q = encodeURIComponent;

export const api = {
  options: () => get("/api/options"),
  projects: () => get("/api/projects"),
  search: (text) => get(`/api/search?q=${q(text)}`),
  project: (id, lang) => get(`/api/projects/${q(id)}` + (lang ? `?lang=${q(lang)}` : "")),
  exports: (id, lang) => get(`/api/projects/${q(id)}/exports?lang=${q(lang)}`),
  jobs: () => get("/api/jobs"),
  estimate: (seconds, lipsync = true) => get(`/api/estimate?seconds=${seconds}&lipsync=${lipsync}`),
  job: (id) => get(`/api/jobs/${id}`),

  dub: (form) => post("/api/dub", form),
  plan: (id, lang, message) => post(`/api/projects/${q(id)}/plan`, { lang, message }),
  apply: (id, lang, plan, lipsync = null) => post(`/api/projects/${q(id)}/apply`, { lang, plan, lipsync }),
  impact: (id, lang, plan, lipsync = null) => post(`/api/projects/${q(id)}/impact`, { lang, plan, lipsync }),
  audition: (id, lang, line, action) => post(`/api/projects/${q(id)}/audition`, { lang, line, action }),
  glossary: (id, lang) => get(`/api/projects/${q(id)}/glossary?lang=${lang}`),
  saveGlossary: (id, lang, entries) => request("PUT", `/api/projects/${q(id)}/glossary`, { lang, entries }),
  subtitled: (id, lang) => post(`/api/projects/${q(id)}/subtitled`, { lang }),
  comments: (id, lang) => get(`/api/projects/${q(id)}/comments?lang=${lang}`),
  addComment: (id, body) => post(`/api/projects/${q(id)}/comments`, body),
  updateComment: (id, cid, body) => post(`/api/projects/${q(id)}/comments/${cid}`, body),
  addLipsync: (id, lang) => post(`/api/projects/${q(id)}/lipsync`, { lang }),
  setLine: (id, lang, line, text) => post(`/api/projects/${q(id)}/lines/${line}`, { lang, text }),
  revoice: (id, lang, voices) => post(`/api/projects/${q(id)}/revoice`, { lang, voices }),

  versions: (id, lang) => get(`/api/projects/${q(id)}/versions?lang=${q(lang)}`),
  probe: (url) => get(`/api/probe?url=${encodeURIComponent(url)}`),
  // Upload a file as the raw body, reporting progress (0-1) as it goes.
  upload(file, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `/api/upload?name=${encodeURIComponent(file.name)}`);
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
      xhr.onload = () => {
        let body = {};
        try {
          body = JSON.parse(xhr.responseText);
        } catch {
          /* not JSON */
        }
        xhr.status < 300 ? resolve(body) : reject(new Error(body.detail || `Upload failed (${xhr.status})`));
      };
      xhr.onerror = () => reject(new Error("The upload was interrupted."));
      xhr.send(file);
    });
  },
  // The user's own voice for a speaker: a recorded Blob or a chosen File.
  myVoice: async (id, speaker, blob) => {
    const res = await fetch(`/api/projects/${q(id)}/myvoice?speaker=${speaker}`, { method: "POST", body: blob });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Upload failed (${res.status})`);
    return data;
  },
  stop: (jobId) => post(`/api/jobs/${jobId}/stop`, {}),
  restore: (id, lang, v) => post(`/api/projects/${q(id)}/versions/${v}/restore`, { lang }),
  jobLines: (jobId) => get(`/api/jobs/${jobId}/lines`),

  previewUrl: (voice, lang) => `/api/voices/preview?voice=${q(voice)}&lang=${q(lang)}`,
  // Follow a job live. Calls onUpdate with every new state; returns a stop function.
  follow(jobId, onUpdate) {
    const source = new EventSource(`/api/jobs/${jobId}/events`);
    source.onmessage = (e) => {
      const job = JSON.parse(e.data);
      onUpdate(job);
      if (["done", "error", "stopped"].includes(job.status)) source.close();
    };
    // If the connection drops (laptop sleeps, tunnel hiccup), poll until it ends.
    source.onerror = () => {
      source.close();
      const poll = async () => {
        try {
          const job = await api.job(jobId);
          onUpdate(job);
          if (!["done", "error", "stopped"].includes(job.status)) setTimeout(poll, 2000);
        } catch {
          setTimeout(poll, 4000);
        }
      };
      setTimeout(poll, 1000);
    };
    return () => source.close();
  },
};
