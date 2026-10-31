// Entry point: the top bar, the current page, the New dub dialog, toasts.

import { createApp } from "/vendor/vue.esm-browser.prod.js";
import { store, boot, go, activeJobs, dismiss, openFromLink } from "./store.js";
import { fmtLeft } from "./util.js";
import Library from "./views/Library.js";
import Project from "./views/Project.js";
import JobPage from "./views/JobPage.js";
import Watch from "./views/Watch.js";
import NewDubDialog from "./components/NewDubDialog.js";
import HelpSheet from "./components/HelpSheet.js";
import Icon from "./components/Icon.js";
import Loco from "./components/Loco.js";

const App = {
  components: { Library, Project, JobPage, Watch, NewDubDialog, HelpSheet },
  data: () => ({ store, ready: false, bootError: "", dark: document.documentElement.dataset.theme === "dark" }),
  computed: {
    route: () => store.route,
    running: activeJobs,
    pillJob() {
      return this.running.find((j) => j.status === "running") || this.running[0];
    },
    pillText() {
      const j = this.pillJob;
      if (!j) return "";
      const stage = j.stages.find((s) => s.key === j.current)?.label;
      const what = { dub: "Dubbing", revoice: "Re-voicing", edit: "Editing", lipsync: "Lip-syncing" }[j.kind];
      if (j.status === "queued") return `${what} · waiting`;
      const left = j.remaining != null ? ` · ${fmtLeft(j.remaining).replace("about ", "~")} left` : "";
      return `${what} · ${stage || "finishing"}${left}`;
    },
  },
  async created() {
    window.addEventListener("keydown", (e) => {
      if (e.key === "?" && !["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) store.help = !store.help;
    });
    try {
      await boot();
      this.ready = true;
      openFromLink();
    } catch (e) {
      this.bootError = e.message;
    }
  },
  methods: {
    go,
    toggleTheme() {
      this.dark = !this.dark;
      document.documentElement.dataset.theme = this.dark ? "dark" : "light";
      try {
        localStorage.setItem("theme", this.dark ? "dark" : "light");
      } catch {
        /* this visit only */
      }
    },
    dismiss,
    openPill() {
      const j = this.pillJob;
      go(j.kind === "dub" ? `#/job/${j.id}` : `#/p/${encodeURIComponent(j.project)}/${j.lang}`);
    },
  },
  template: `
    <header class="topbar">
      <a class="brand" href="#/">
        <span class="brand-mark" aria-hidden="true">
          <svg viewBox="-20 -20 40 40">
            <g class="petal-ring" fill="currentColor" opacity=".55">
              <ellipse v-for="k in 12" :key="k" rx="2.6" ry="5.5" cy="-13" :transform="'rotate(' + k * 30 + ')'" />
              <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="24s" repeatCount="indefinite" />
            </g>
            <circle r="9.5" fill="currentColor" />
            <path d="M-5.5 -2v4M-2.75 -4.5v9M0 -3v6M2.75 -5.5v11M5.5 -2v4" stroke="#1e0f00" stroke-width="1.7" stroke-linecap="round" />
          </svg>
        </span><span class="brand-name"><b>Dubbing Studio</b></span></a>
      <span class="spacer"></span>
      <button v-if="pillJob && route.name !== 'watch'" class="job-pill" @click="openPill" title="Open the running job">
        <span class="spinner"></span><b>{{ pillText }}</b>
        <span v-if="running.length > 1">+{{ running.length - 1 }}</span>
      </button>
      <button class="btn ghost icon" :title="dark ? 'Light mode' : 'Dark mode'" :aria-label="dark ? 'Light mode' : 'Dark mode'" @click="toggleTheme">{{ dark ? '☀' : '☾' }}</button>
      <button v-if="route.name !== 'watch'" class="btn ghost icon" title="How do I…? (press ?)" @click="store.help = true">?</button>
      <!-- The home page has its own link box and language picker. -->
      <button v-if="route.name !== 'watch' && route.name !== 'library'" class="btn primary" :disabled="!ready" @click="store.newDub = {}">+ New dub</button>
    </header>

    <div v-if="bootError" class="page"><div class="note error">Cannot reach the studio server: {{ bootError }}</div></div>
    <template v-else-if="ready">
      <Library v-if="route.name === 'library'" />
      <Project v-else-if="route.name === 'project'" :id="route.id" :lang="route.lang" :key="route.id" />
      <JobPage v-else-if="route.name === 'job'" :id="route.id" :key="route.id" />
      <Watch v-else-if="route.name === 'watch'" :id="route.id" :lang="route.lang" :key="'w' + route.id" />
    </template>

    <NewDubDialog v-if="store.newDub" :presets="store.newDub" @close="store.newDub = null" />
    <HelpSheet v-if="store.help" @close="store.help = false" />

    <div class="toasts">
      <div v-for="t in store.toasts" :key="t.id" class="toast" :class="t.kind">
        <span>{{ t.text }}</span><span class="x" @click="dismiss(t.id)">✕</span>
      </div>
    </div>`,
};

const app = createApp(App);
// go() and projectPath() are used directly in templates.
app.config.globalProperties.go = go;
// Line icons and the Dub Express locomotive, used on every page.
app.component("Icon", Icon);
app.component("Loco", Loco);

// v-reveal: the element fades and rises in the first time it scrolls into
// view (studio.css .reveal). A number value staggers it, e.g. v-reveal="i".
const seen = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      e.target.classList.add("in");
      seen.unobserve(e.target);
    }
  },
  { threshold: 0.12 },
);
app.directive("reveal", {
  mounted(el, { value }) {
    el.classList.add("reveal");
    if (typeof value === "number") el.style.setProperty("--d", Math.min(value, 8) * 70 + "ms");
    seen.observe(el);
  },
  unmounted(el) {
    seen.unobserve(el);
  },
});
app.mount("#app");
