// Home page. On top, the Indic / International switch and the language
// picker: in Indic mode components/BhashaStage.js (a 3D
// lotus with every language orbiting it, or a dot map of India); in
// International mode components/GlobeStage.js (a turning globe). Click a
// language to dub into it. Below, the dubs made so far, as
// a grid of cards, newest first and grouped by day, with the ones being made
// right now at the top.
//
// The finder searches titles, channels and language names, and — through the
// server — every line said in every dub, in the original and the dub. A hit
// on a line opens that dub at that line.

import { api } from "../api.js";
import { store, activeJobs, go, projectPath } from "../store.js";
import { fmtTime, ago, youtubeId, fmtLeft, splitLangLabel, glyph, STATIONS, ENGINE, STAGE_WORDS } from "../util.js";
import BhashaStage from "../components/BhashaStage.js";
import GlobeStage from "../components/GlobeStage.js";
import DubExpress from "../components/DubExpress.js";

const DAY = 86400;
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

export default {
  components: { BhashaStage, GlobeStage, DubExpress },
  data: () => ({
    store,
    projects: [],
    loading: true,
    error: "",
    query: "",
    only: null, // language code the thread is filtered to
    said: [], // line hits from /api/search
    searching: false,
    url: "",
    word: 0,
    // "indic": the lotus / India map and the Indian languages;
    // "international": a turning globe and the world languages.
    mode: (() => {
      try {
        return localStorage.getItem("home-mode") || "indic";
      } catch {
        return "indic";
      }
    })(),
    dragOver: false,
    hood: false,
    searchTimer: 0,
    STATIONS, // stop names for the dubs being made (util.js)
    shownStats: { dubs: 0, langs: 0, mins: 0 }, // counted up from 0 when the page opens
  }),
  computed: {
    running: activeJobs,
    allLangs() {
      return [...(store.options?.languages || []), ...(store.options?.international || [])].map((l) => ({
        code: l.code,
        ...splitLangLabel(l.label),
        glyph: glyph(l.label),
      }));
    },
    langs() {
      const list = this.mode === "international" ? store.options?.international : store.options?.languages;
      return (list || []).map((l) => ({ code: l.code, ...splitLangLabel(l.label), glyph: glyph(l.label) }));
    },
    active() {
      return this.langs[this.word % Math.max(1, this.langs.length)] || null;
    },
    pastedId() {
      return youtubeId(this.url);
    },
    q() {
      return this.query.trim().toLowerCase();
    },
    // Typing a link into the search box is a request to dub it.
    queryIsLink() {
      return /^https?:\/\/\S+\.\S+/i.test(this.query.trim());
    },
    // Languages that have at least one dub, for the filter beads.
    usedLangs() {
      const n = {};
      for (const p of this.projects) for (const l of p.langs) n[l.code] = (n[l.code] || 0) + 1;
      return this.langs.filter((l) => n[l.code]).map((l) => ({ ...l, count: n[l.code] }));
    },
    // {project id: [line hits]} from the server search, for the language filter.
    saidBy() {
      const out = {};
      for (const h of this.said) {
        if (this.only && h.lang !== this.only) continue;
        (out[h.id] = out[h.id] || []).push(h);
      }
      return out;
    },
    shown() {
      const q = this.queryIsLink ? "" : this.q;
      return this.projects.filter((p) => {
        if (this.only && !p.langs.some((l) => l.code === this.only)) return false;
        if (!q) return true;
        const names = p.langs.map((l) => `${l.name} ${this.nativeOf(l.code)}`).join(" ");
        return `${p.title} ${p.channel} ${names}`.toLowerCase().includes(q) || this.saidBy[p.id];
      });
    },
    // Dubs grouped by when they were last changed: today, yesterday, earlier.
    groups() {
      const midnight = new Date().setHours(0, 0, 0, 0) / 1000;
      const out = [
        { key: "today", label: "Today", native: "आज", items: [] },
        { key: "yesterday", label: "Yesterday", native: "कल", items: [] },
        { key: "week", label: "This week", native: "इस हफ़्ते", items: [] },
        { key: "earlier", label: "Earlier", native: "पहले", items: [] },
      ];
      for (const p of this.shown) {
        const g = p.updated >= midnight ? 0 : p.updated >= midnight - DAY ? 1 : p.updated >= midnight - 6 * DAY ? 2 : 3;
        out[g].items.push(p);
      }
      return out.filter((g) => g.items.length);
    },
    stats() {
      const langs = new Set(this.projects.flatMap((p) => p.langs.map((l) => l.code)));
      const secs = this.projects.reduce((t, p) => t + (p.duration || 0) * p.langs.length, 0);
      return { dubs: this.projects.reduce((t, p) => t + p.langs.length, 0), langs: langs.size, mins: Math.round(secs / 60) };
    },
    filtering() {
      return !!(this.q || this.only);
    },
    hoodStages() {
      return Object.keys(ENGINE).map((k) => ({ key: k, ...STATIONS[k], words: STAGE_WORDS[k], engine: ENGINE[k] }));
    },
  },
  watch: {
    "store.finished"() {
      this.load();
    },
    // The little numbers under "Your dubs" count up once, quickly.
    stats(to) {
      const from = { ...this.shownStats };
      const t0 = performance.now();
      const step = (now) => {
        const t = Math.min(1, (now - t0) / 900);
        const k = 1 - (1 - t) ** 3;
        for (const key of Object.keys(to)) this.shownStats[key] = Math.round(from[key] + (to[key] - from[key]) * k);
        if (t < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    },
    query() {
      clearTimeout(this.searchTimer);
      if (this.q.length < 2 || this.queryIsLink) {
        this.said = [];
        this.searching = false;
        return;
      }
      this.searching = true;
      this.searchTimer = setTimeout(this.searchLines, 250);
    },
  },
  created() {
    this.load();
    this.onKey = (e) => {
      if (e.key === "/" && !["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) {
        e.preventDefault();
        this.$refs.find?.focus();
      }
    };
    window.addEventListener("keydown", this.onKey);
  },
  unmounted() {
    window.removeEventListener("keydown", this.onKey);
    clearTimeout(this.searchTimer);
  },
  methods: {
    fmtTime,
    ago,
    fmtLeft,
    async load() {
      try {
        this.projects = await api.projects();
        this.error = "";
      } catch (e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },
    async searchLines() {
      const asked = this.q;
      try {
        const hits = await api.search(asked);
        if (asked === this.q) this.said = hits;
      } catch {
        this.said = [];
      } finally {
        if (asked === this.q) this.searching = false;
      }
    },
    open(p, lang) {
      const filtered = this.only && p.langs.some((l) => l.code === this.only) ? this.only : null;
      go(projectPath(p.id, lang || filtered || p.langs[0].code));
    },
    openLine(h) {
      store.jumpTo = { id: h.id, lang: h.lang, line: h.line };
      go(projectPath(h.id, h.lang));
    },
    // The search words marked in a piece of text (escaped first).
    mark(text) {
      const safe = esc(text || "");
      if (!this.q || this.queryIsLink) return safe;
      const re = new RegExp(esc(this.q).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
      return safe.replace(re, (m) => `<mark>${m}</mark>`);
    },
    clearFind() {
      this.query = "";
      this.only = null;
    },
    // A dub's language badge, from every language (a Hindi dub keeps "हि"
    // when the page is in International mode).
    glyphOf(code) {
      return this.allLangs.find((l) => l.code === code)?.glyph || code;
    },
    nativeOf(code) {
      return this.allLangs.find((l) => l.code === code)?.native || code;
    },
    progress(j) {
      return Math.round((100 * Object.keys(j.timings).length) / j.stages.length);
    },
    stageLabel(j) {
      if (j.status === "queued") return "Waiting its turn";
      if (j.stopping) return "Stopping…";
      return STAGE_WORDS[j.current] || j.stages.find((s) => s.key === j.current)?.label || "Finishing";
    },
    // No local frame: use YouTube's own thumbnail, else a plain arch.
    thumbFallback(e, p) {
      const v = youtubeId(p.url) || p.id.slice(0, 11);
      if (!e.target.dataset.tried) {
        e.target.dataset.tried = "1";
        e.target.src = `https://i.ytimg.com/vi/${v}/mqdefault.jpg`;
      } else e.target.style.visibility = "hidden";
    },
    // A language picked on the stage opens the New dub dialog with it chosen.
    setMode(m) {
      this.mode = m;
      this.word = 0;
      document.documentElement.dataset.mode = m; // the page's look follows (studio.css)
      try {
        localStorage.setItem("home-mode", m);
      } catch {
        /* this visit only */
      }
    },
    pick(l) {
      store.newDub = { lang: l.code, mode: this.mode, ...(this.url.trim() ? { url: this.url.trim() } : {}) };
      this.url = "";
    },
    startFromLink(link = this.url) {
      store.newDub = { url: link.trim(), mode: this.mode };
      this.url = "";
    },
    // A file dropped (or chosen) on the welcome box: the dialog uploads it.
    onFile(file) {
      if (file) store.newDub = { file };
    },
    onPaste() {
      this.$nextTick(() => this.pastedId && this.startFromLink());
    },
  },
  template: `
    <main class="page home">
      <section class="hero" :class="{ over: dragOver, intl: mode === 'international' }"
               @dragover.prevent="dragOver = true" @dragleave="dragOver = false" @drop.prevent="dragOver = false; onFile($event.dataTransfer.files[0])">
        <!-- Behind the words: a mandala in line art and a block-print border. -->
        <div class="hero-art" aria-hidden="true">
          <svg class="mandala" viewBox="-100 -100 200 200">
            <circle r="18" /><circle r="30" class="gold" /><circle r="58" /><circle r="88" class="gold" /><circle r="96" class="thin" />
            <ellipse v-for="k in 12" :key="'p' + k" rx="9" ry="22" cy="-44" :transform="'rotate(' + k * 30 + ')'" />
            <ellipse v-for="k in 12" :key="'q' + k" rx="6" ry="14" cy="-44" class="gold" :transform="'rotate(' + (k * 30 + 15) + ')'" />
            <path v-for="k in 24" :key="'l' + k" d="M0 -60 Q7 -74 0 -86 Q-7 -74 0 -60Z" :transform="'rotate(' + k * 15 + ')'" />
            <circle v-for="k in 36" :key="'d' + k" r="1.6" cy="-92" class="dot" :transform="'rotate(' + k * 10 + ')'" />
          </svg>
          <div class="print-border"></div>
        </div>
        <div class="hero-copy">
          <div class="stage-mode" role="radiogroup" aria-label="Indic or International">
            <button role="radio" :aria-checked="mode === 'indic'" :class="{ on: mode === 'indic' }" @click="setMode('indic')">Indic</button>
            <button role="radio" :aria-checked="mode === 'international'" :class="{ on: mode === 'international' }" @click="setMode('international')">International</button>
          </div>
          <h1>Make any video speak<br /><span class="word" :class="{ rtl: active && ['ur', 'ks', 'sd'].includes(active.code) }" :key="word" :lang="active && active.code">{{ active ? active.native : 'your language' }}</span></h1>
          <p>Paste a link or drop a file. Every person keeps a voice that sounds like them — in the language you pick — and their lips move with the new words.</p>
          <div class="paste">
            <span class="paste-icon" aria-hidden="true"><Icon name="link" /></span>
            <input v-model="url" placeholder="Paste a video link — YouTube, Instagram, Vimeo, X…" aria-label="Video link"
                   @keydown.enter="url.trim() && startFromLink()" @paste="onPaste" />
            <button class="btn primary lg" :disabled="!url.trim()" @click="startFromLink()">Dub it →</button>
          </div>
          <div class="or-upload">or <a href="#" @click.prevent="$refs.file.click()">upload a video or audio file</a> — or just drop it anywhere here
            <input ref="file" type="file" accept="video/*,audio/*,.mkv,.mov" hidden @change="onFile($event.target.files[0])" /></div>
          <ol class="steps3">
            <li><b>{{ mode === 'international' ? '1' : '१' }}</b> Paste a link</li>
            <li><b>{{ mode === 'international' ? '2' : '२' }}</b> {{ mode === 'international' ? 'Pick a language' : 'Pick a bhasha' }}</li>
            <li><b>{{ mode === 'international' ? '3' : '३' }}</b> Watch, then tweak anything</li>
          </ol>
        </div>
        <BhashaStage v-if="mode === 'indic'" :langs="langs" :active="word" @active="word = $event" @pick="pick" />
        <GlobeStage v-else :langs="langs" :active="word" @active="word = $event" @pick="pick" />
      </section>

      <section v-if="mode === 'indic'" class="hood" :class="{ open: hood }" v-reveal>
        <button class="hood-head" @click="hood = !hood" :aria-expanded="hood">
          <span class="hood-title">How a dub is made <span class="hint">— the Dub Express: ten stops, north to south, Jammu Tawi to Kanyakumari</span></span>
          <span>{{ hood ? '▴' : '▾' }}</span>
        </button>
        <DubExpress v-if="hood" :stages="hoodStages" />
        <p v-if="hood" class="hood-foot hint">Click anywhere for the <b>next stop</b>. Changing a few lines later runs an <b>express</b>: it stops only where those lines need it.</p>
      </section>

      <div v-if="error" class="note error">Could not load your dubs: {{ error }}</div>

      <section v-if="projects.length || running.length" class="dubs-wrap">
        <div class="dubs-head" v-reveal>
          <h2>Your dubs <span v-if="mode === 'indic'" class="native">आपके डब</span></h2>
          <p class="stats" v-if="projects.length">
            <span><b>{{ shownStats.dubs }}</b> dubs</span><span><b>{{ shownStats.langs }}</b> languages</span><span><b>{{ shownStats.mins }}</b> min of speech</span>
          </p>
        </div>

        <div class="finder" v-reveal="1">
          <div class="find-box">
            <span class="find-icon" aria-hidden="true"><Icon name="search" /></span>
            <input ref="find" v-model="query" aria-label="Search your dubs"
                   placeholder="Find a dub — title, language, or words someone says"
                   @keydown.esc="clearFind(); $event.target.blur()" />
            <span v-if="searching" class="spinner"></span>
            <button v-if="filtering" class="btn ghost sm" @click="clearFind">Clear</button>
            <kbd v-else title="Press / to search">/</kbd>
          </div>
          <div class="find-langs" v-if="usedLangs.length > 1">
            <button class="fl" :class="{ on: !only }" @click="only = null">All</button>
            <button v-for="l in usedLangs" :key="l.code" class="fl" :class="{ on: only === l.code }" :lang="l.code"
                    @click="only = only === l.code ? null : l.code"><b>{{ l.glyph }}</b>{{ l.en }}<i>{{ l.count }}</i></button>
          </div>
          <div v-if="queryIsLink" class="find-link">
            That's a link. <button class="btn primary sm" @click="startFromLink(query); query = ''">Dub this video →</button>
          </div>
          <div v-else-if="filtering" class="find-sum" aria-live="polite">
            <template v-if="shown.length">Showing <b>{{ shown.length }}</b> of {{ projects.length }}<template v-if="said.length"> · <b>{{ said.length }}</b> line{{ said.length > 1 ? 's' : '' }} where someone says “{{ query.trim() }}”</template></template>
            <template v-else-if="!searching">Nothing matches. Try a word someone says, a language like <i>Tamil</i> or <i>தமிழ்</i>, or part of a title.</template>
          </div>
        </div>

        <div v-if="running.length && !filtering" class="day"><span>Being made<template v-if="mode === 'indic'"> · <i>बन रहा है</i></template></span></div>
        <div v-if="running.length && !filtering" class="dub-grid">
          <article v-for="j in running" :key="'job' + j.id" class="dub-card live" @click="go('#/job/' + j.id)">
            <div class="dc-thumb dc-making">
              <div class="ring" :style="{ '--p': progress(j) }"><Loco :size="34" /></div>
              <span class="badge">{{ progress(j) }}%</span>
            </div>
            <div class="dc-body">
              <div class="dc-title">{{ j.kind === 'dub' ? 'A new dub is being made' : j.title }}</div>
              <div class="dc-meta"><span>{{ stageLabel(j) }}<template v-if="STATIONS[j.current]"> · near {{ STATIONS[j.current].place }}</template></span></div>
              <div class="dc-langs"><span class="lbead" :title="j.lang">{{ glyphOf(j.lang) }}</span><span class="hint">{{ nativeOf(j.lang) }}<template v-if="j.remaining != null"> · {{ fmtLeft(j.remaining) }} left</template></span></div>
            </div>
          </article>
        </div>

        <template v-for="g in groups" :key="g.key">
          <div class="day"><span>{{ g.label }}<template v-if="mode === 'indic'"> · <i>{{ g.native }}</i></template></span></div>
          <div class="dub-grid">
            <article v-for="(p, i) in g.items" :key="p.id" v-reveal="i" class="dub-card" @click="open(p)">
              <div class="dc-thumb">
                <img :src="'/media/' + p.id + '/thumb.jpg'" loading="lazy" alt="" @error="thumbFallback($event, p)" />
                <span v-if="p.duration" class="badge">{{ fmtTime(p.duration) }}</span>
              </div>
              <div class="dc-body">
                <div class="dc-title" v-html="mark(p.title)"></div>
                <div class="dc-meta"><span class="ellipsis" v-html="mark(p.channel || (p.url.startsWith('upload:') ? 'Your upload' : 'Web video'))"></span><span style="flex:none">· {{ ago(p.updated) }}</span></div>
                <div class="dc-langs">
                  <button v-for="l in p.langs" :key="l.code" class="lbead" :class="{ dim: only && only !== l.code }" :title="'Open the ' + l.name + ' dub'" :lang="l.code" @click.stop="open(p, l.code)">{{ glyphOf(l.code) }}</button>
                  <span class="hint ellipsis">{{ p.langs.map(l => l.name).join(' · ') }}</span>
                </div>
                <ul v-if="saidBy[p.id]" class="said">
                  <li v-for="h in saidBy[p.id].slice(0, 3)" :key="h.lang + h.line" @click.stop="openLine(h)" :title="'Open the dub at this line'">
                    <span class="said-t">{{ fmtTime(h.start) }}</span><span class="lbead xs" :lang="h.lang">{{ glyphOf(h.lang) }}</span>
                    <span class="said-text"><span v-html="mark(h.text)"></span><span class="said-dub" :lang="h.lang" v-html="mark(h.dub)"></span></span>
                    <span class="said-go">▶</span>
                  </li>
                  <li v-if="saidBy[p.id].length > 3" class="said-more">+ {{ saidBy[p.id].length - 3 }} more</li>
                </ul>
              </div>
            </article>
          </div>
        </template>
      </section>

      <div v-if="!loading && !projects.length && !running.length" class="empty">
        <div class="diya" aria-hidden="true"><Icon name="lamp" :size="44" /></div>
        <h2>Light the first lamp</h2>
        <p>Your first dub is one link away. Short videos (a minute or two) are ready fastest — great for a first try.</p>
      </div>
    </main>`,
};
