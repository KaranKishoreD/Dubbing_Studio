// State shared by every page: form options, running jobs, notifications,
// and a tiny hash router (#/, #/p/<project>/<lang>, #/job/<id>).

import { reactive } from "/vendor/vue.esm-browser.prod.js";
import { api } from "./api.js";

export const store = reactive({
  options: null,
  jobs: {}, // id -> latest job state
  finished: null, // the last job that ended; pages watch it to refresh
  toasts: [],
  route: parseRoute(),
  newDub: null, // non-null = the New dub dialog is open, with these presets
  celebrate: false, // a dub just finished: the studio shows a little confetti
  help: false, // the help sheet is open
  jumpTo: null, // {id, lang, line}: a search hit; the dub opens at that line
});

// ── Routing ──────────────────────────────────────────────────────────────────
function parseRoute() {
  const [, page = "", a = "", b = ""] = location.hash.split("/").map(decodeURIComponent);
  if (page === "p" && a) return { name: "project", id: a, lang: b || null };
  if (page === "job" && a) return { name: "job", id: Number(a) };
  if (page === "watch" && a) return { name: "watch", id: a, lang: b || null };
  return { name: "library" };
}
// "#/new" is a shareable link to the New dub dialog, over the library.
function onHash() {
  if (location.hash === "#/new") {
    store.newDub = {};
    history.replaceState(null, "", "#/");
  }
  store.route = parseRoute();
}
window.addEventListener("hashchange", onHash);
export const openFromLink = onHash;

export function go(path) {
  location.hash = path;
}
export const projectPath = (id, lang) => `#/p/${encodeURIComponent(id)}${lang ? "/" + lang : ""}`;

// ── Notifications ────────────────────────────────────────────────────────────
let toastId = 0;
export function toast(text, kind = "ok", ms = 5000) {
  const id = ++toastId;
  store.toasts.push({ id, text, kind });
  if (ms) setTimeout(() => dismiss(id), ms);
}
export function dismiss(id) {
  store.toasts = store.toasts.filter((t) => t.id !== id);
}

// ── Jobs ─────────────────────────────────────────────────────────────────────
// Every job started from this page, or already running when it opened, is
// followed live. When one ends it is announced and `finished` is set.
export function track(job) {
  store.jobs[job.id] = job;
  api.follow(job.id, (update) => {
    const before = store.jobs[job.id];
    store.jobs[job.id] = update;
    const ended = ["done", "error", "stopped"].includes(update.status);
    if (ended && before?.status !== update.status) {
      store.finished = update;
      if (update.status === "done") toast(doneText(update), "ok", 7000);
      else if (update.status === "stopped") toast(update.message, "ok", 7000);
      else toast(`${update.title}: ${update.message}`, "error", 0);
    }
  });
}

function doneText(job) {
  const what = { dub: "Dub ready", revoice: "New voices ready", edit: "Changes applied", lipsync: "Lip sync added" }[job.kind];
  return `${what} — ${job.title.length > 60 ? job.title.slice(0, 57) + "…" : job.title}`;
}

export const activeJobs = () =>
  Object.values(store.jobs)
    .filter((j) => j.status === "queued" || j.status === "running")
    .sort((a, b) => a.id - b.id);

export const jobFor = (projectId) => activeJobs().find((j) => j.project === projectId);

export async function boot() {
  const [options, running] = await Promise.all([api.options(), api.jobs()]);
  store.options = options;
  running.forEach(track);
}
