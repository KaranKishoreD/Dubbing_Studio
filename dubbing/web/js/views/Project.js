// One dubbed video: the player and timeline on the left, and on the right
// the Lines, Voices, Ask, Versions and Download tabs.
//
// Lines can be selected (drag on the timeline, or tick them in the script);
// the selection bar then offers changes for just those lines.
//
// Changes are collected first (PendingBar) and applied together in one job,
// so several fixes cost one re-dub. The list survives a page reload.

import { api } from "../api.js";
import { store, go, projectPath, track, toast, jobFor } from "../store.js";
import Player from "../components/Player.js";
import LinesPanel from "../components/LinesPanel.js";
import VoicesPanel from "../components/VoicesPanel.js";
import AskPanel from "../components/AskPanel.js";
import VersionsPanel from "../components/VersionsPanel.js";
import DownloadPanel from "../components/DownloadPanel.js";
import SelectionBar from "../components/SelectionBar.js";
import JobProgress from "../components/JobProgress.js";
import PendingBar from "../components/PendingBar.js";
import CommentsCard from "../components/CommentsCard.js";
import EditGuide from "../components/EditGuide.js";

const stored = (key) => {
  try {
    return JSON.parse(localStorage.getItem(key)) || [];
  } catch {
    return [];
  }
};

// [key, label, what it is for]
const TABS = [
  ["script", "Lines", "Every line: watch it, rewrite it, fix the transcript"],
  ["cast", "Voices", "Who speaks, and with which voice"],
  ["revise", "Ask", "Describe changes in your own words; names & words; comments"],
  ["history", "Versions", "Every version of this dub — undo or go back"],
  ["export", "Download", "The dubbed video, soundtrack and subtitles"],
];

export default {
  components: { Player, LinesPanel, VoicesPanel, AskPanel, VersionsPanel, DownloadPanel, SelectionBar, JobProgress, PendingBar, CommentsCard, EditGuide },
  props: { id: { type: String, required: true }, lang: { type: String, default: null } },
  data: () => ({
    store,
    project: null,
    error: "",
    tab: "script",
    now: 0,
    clicked: null, // the line last clicked, shown as active while paused
    picked: [], // selected line ids
    lastPicked: null, // for shift-click ranges
    lastChange: "",
    ask: null, // a request for the Ask tab to plan right away
    previewing: null, // an older version being watched: {v, video}
    pending: [], // changes waiting to be applied: [{request, actions}]
    versions: [], // newest first (server/versions.py)
    comments: [], // review comments (server/comments.py)
    confetti: [], // a dub just finished: a short burst of confetti
    guide: false, // the first-visit guide is shown
    undoing: false,
    editTips: (() => {
      try {
        return localStorage.getItem("edit-tips") !== "hidden";
      } catch {
        return true;
      }
    })(), // the "What would you like to change?" choices are shown
    TABS,
  }),
  computed: {
    // Where the person is in pick → choose → apply.
    editStep() {
      if (this.pending.length) return 3;
      return this.pickedLines.length ? 2 : 1;
    },
    job() {
      return jobFor(this.id);
    },
    busy() {
      return !!this.job;
    },
    activeId() {
      const l = this.project?.lines.find((l) => this.now >= l.start && this.now < l.end);
      return l ? l.id : this.clicked;
    },
    pickedLines() {
      return (this.project?.lines || []).filter((l) => this.picked.includes(l.id));
    },
    pickedSpan() {
      const l = this.pickedLines;
      return l.length ? [Math.min(...l.map((x) => x.start)), Math.max(...l.map((x) => x.end))] : null;
    },
    openComments() {
      return this.comments.filter((c) => !c.done);
    },
    // Lines the latest change redid: they glow on the timeline.
    lastChanged() {
      const v = this.versions[0];
      if (!v || v.kind === "dub" || !v.changes) return [];
      return [...v.changes.matchAll(/\*\*S(\d+)\*\*/g)].map((m) => Number(m[1]));
    },
    pendingKey() {
      return `pending:${this.id}:${this.lang || ""}`;
    },
    // Lines a waiting change will touch, marked in the script.
    pendingIds() {
      const ids = new Set();
      for (const r of this.pending)
        for (const a of r.actions) {
          (a.lines || []).forEach((i) => ids.add(i));
          if (a.line != null) ids.add(a.line);
        }
      return [...ids];
    },
    // New people added by waiting changes, so more lines can be given to them.
    newPeople() {
      const out = [];
      this.pending.forEach((r, i) =>
        r.actions.forEach((a) => {
          if (a.type === "new_speaker") out.push({ idx: i, gender: a.gender, label: `New ${a.gender === "female" ? "woman" : "man"} ${out.length + 1}` });
        }),
      );
      return out;
    },
    media() {
      // While watching an older version, it replaces the current dub.
      const m = this.project.media;
      return this.previewing ? { ...m, dubbed: this.previewing.video } : m;
    },
  },
  watch: {
    id: "load",
    lang() {
      this.previewing = null;
      this.picked = [];
      this.load();
    },
    pendingKey: {
      immediate: true,
      handler(key) {
        this.pending = stored(key);
      },
    },
    "store.finished"(job) {
      if (job?.project !== this.id) return;
      if (job.status === "stopped") {
        // Put the stopped changes back in the list, to correct and apply again.
        if (job.plan) {
          const back = job.plan.requests.filter((r) => r.actions?.length).map((r) => ({ request: r.request, actions: r.actions }));
          this.pending = [...back, ...this.pending];
          this.savePending();
          toast("Stopped. Your changes are back in the list — adjust them and apply again.", "ok", 7000);
        }
        return;
      }
      if (job.kind === "edit" && job.status === "done") this.lastChange = job.message;
      this.previewing = null;
      this.load();
    },
  },
  created() {
    this.load();
    try {
      this.guide = !localStorage.getItem("guide-seen");
    } catch {
      this.guide = false;
    }
    if (store.celebrate) {
      store.celebrate = false;
      const colors = ["#ff9a1f", "#ffb627", "#f5c84c", "#ec3b72", "#ff6f3c"]; // marigold and rose petals
      this.confetti = Array.from({ length: 70 }, (_, i) => ({
        left: Math.random() * 100,
        delay: Math.random() * 0.6,
        color: colors[i % colors.length],
        spin: Math.random() * 720 - 360,
      }));
      toast("Your dub is ready! Press play — or pick a moment below to change anything.", "ok", 7000);
      setTimeout(() => (this.confetti = []), 3500);
    }
  },
  methods: {
    projectPath,
    async load() {
      try {
        const p = await api.project(this.id, this.lang);
        this.project = p;
        this.error = "";
        const ids = new Set(p.lines.map((l) => l.id));
        this.picked = this.picked.filter((id) => ids.has(id));
        if (!this.lang) history.replaceState(null, "", projectPath(p.id, p.lang));
        this.loadVersions();
        this.loadComments();
        this.jump();
      } catch (e) {
        this.error = e.message;
      }
    },
    // Opened from a search hit on the home page: start at that line.
    jump() {
      const j = store.jumpTo;
      if (!j || j.id !== this.id || j.lang !== this.project.lang) return;
      store.jumpTo = null;
      const line = this.project.lines.find((l) => l.id === j.line);
      if (line) this.$nextTick(() => this.$refs.player && this.seekTo(line));
    },
    setEditTips(on) {
      this.editTips = on;
      try {
        localStorage.setItem("edit-tips", on ? "shown" : "hidden");
      } catch {
        /* this visit only */
      }
    },
    // A choice in the edit guide: go where it is done and show what to do.
    goEdit(key) {
      const flash = (el, cls = "flash") => {
        if (!el) return;
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        el.classList.remove(cls);
        void el.offsetWidth; // restart the animation
        el.classList.add(cls);
        setTimeout(() => el.classList.remove(cls), 3200);
      };
      const side = () => this.$el.querySelector(".side");
      if (key === "moment") {
        flash(this.$el.querySelector(".timeline"));
        toast("Drag across the film strip (or tick lines in Lines) to pick the part you want to change.", "ok", 6000);
      } else if (key === "line") {
        this.tab = "script";
        this.$nextTick(() => flash(side(), "hint-edit"));
        toast("Click ✎ Dub on a line to rewrite it, or ✎ Transcript to fix what was heard.", "ok", 6000);
      } else if (key === "voice") {
        this.tab = "cast";
        this.$nextTick(() => flash(this.$el.querySelector(".tab-body")));
        toast("Pick a new voice for anyone below — then Apply.", "ok", 5000);
      } else if (key === "names") {
        this.tab = "revise";
        this.$nextTick(() => {
          const head = this.$el.querySelector(".gl-head");
          if (head && !this.$el.querySelector(".gl-body")) head.click();
          flash(this.$el.querySelector(".glossary"));
        });
      } else if (key === "ask") {
        this.tab = "revise";
        this.$nextTick(() => {
          const box = this.$el.querySelector(".side textarea");
          flash(box?.closest(".field"));
          box?.focus();
        });
      } else if (key === "lips") {
        this.applyPlan({ requests: [{ request: "Lip sync the whole video again", actions: [{ type: "lip_sync" }] }] });
        toast("Added to your changes: lip sync for the whole video. Press Apply to run it.", "ok", 6000);
      } else if (key === "undo") {
        this.tab = "history";
      }
    },
    closeGuide() {
      this.guide = false;
      try {
        localStorage.setItem("guide-seen", "1");
      } catch {
        /* shown again next time */
      }
    },
    async loadComments() {
      try {
        this.comments = await api.comments(this.id, this.project.lang);
      } catch {
        this.comments = [];
      }
    },
    // A comment becomes an editable request in the Ask tab for the lines at its moment.
    sendComment(c) {
      const a = c.t - 0.3;
      const b = (c.t2 || c.t) + 0.3;
      let ids = this.project.lines.filter((l) => l.end > a && l.start < b).map((l) => l.id);
      if (!ids.length) {
        const near = [...this.project.lines].sort((x, y) => Math.abs(x.start - c.t) - Math.abs(y.start - c.t))[0];
        ids = near ? [near.id] : [];
      }
      this.picked = ids;
      this.tab = "revise";
      this.ask = { text: c.text, edit: true };
      toast(`${c.name}'s comment is in the Ask box — edit it, then Plan.`, "ok", 4000);
    },
    openComment(c) {
      this.$refs.player.seek(c.t, false);
      this.tab = "revise";
    },
    async loadVersions() {
      try {
        this.versions = await api.versions(this.id, this.project.lang);
      } catch {
        this.versions = [];
      }
    },
    // Undo = restore the version before the current one (itself saved as a
    // version, so the undone change can be brought back from History).
    async undo() {
      const prev = this.versions[1];
      if (!prev || !confirm(`Undo “${this.versions[0].title}”? It stays in History, so you can bring it back.`)) return;
      this.undoing = true;
      try {
        await api.restore(this.id, this.project.lang, prev.v);
        toast(`Undone — back to version ${prev.v}.`);
        this.previewing = null;
        await this.load();
      } catch (e) {
        toast(e.message, "error", 6000);
      } finally {
        this.undoing = false;
      }
    },
    preview(v) {
      this.previewing = v;
      if (v?.at != null) {
        // The older video loads first; then jump to the change.
        const at = v.at;
        setTimeout(() => this.$refs.player.seek(at, true), 400);
      }
    },
    seekTime(t) {
      this.$refs.player.seek(t, true);
    },
    seekTo(line) {
      this.clicked = line.id;
      this.$refs.player.seek(line.start);
    },
    playLine(line, which) {
      this.clicked = line.id;
      this.$refs.player.playSpan(line.start, line.end, which);
    },

    // ── Selection ────────────────────────────────────────────────────────
    toggle(id, shift) {
      const order = this.project.lines.map((l) => l.id);
      if (shift && this.lastPicked != null) {
        const [a, b] = [order.indexOf(this.lastPicked), order.indexOf(id)].sort((x, y) => x - y);
        this.picked = [...new Set([...this.picked, ...order.slice(a, b + 1)])];
      } else {
        this.picked = this.picked.includes(id) ? this.picked.filter((x) => x !== id) : [...this.picked, id];
      }
      this.lastPicked = id;
    },
    selectRange(r) {
      if (!r) return (this.picked = []);
      const [a, b] = r;
      if (!this.project.editable) return;
      this.picked = this.project.lines.filter((l) => l.end > a && l.start < b).map((l) => l.id);
      if (!this.picked.length) toast("No lines in that stretch.", "error", 3000);
    },
    describe(text) {
      this.tab = "revise";
      this.ask = { text };
    },

    // ── Changes ──────────────────────────────────────────────────────────
    async startJob(promise, label) {
      try {
        track(await promise);
        toast(label);
        return true;
      } catch (e) {
        toast(e.message, "error", 8000);
        return false;
      }
    },
    // ── Waiting changes ──────────────────────────────────────────────────
    savePending() {
      try {
        localStorage.setItem(this.pendingKey, JSON.stringify(this.pending));
      } catch {
        /* the list just won't survive a reload */
      }
    },
    // Add a plan's requests to the waiting list. A later choice for the same
    // thing (a speaker's voice, a line's wording) replaces the earlier one.
    queue(plan) {
      for (const r of plan.requests || []) {
        const acts = r.actions || [];
        if (acts.some((a) => a.type === "redetect_speakers")) {
          // Re-detecting replaces the whole dub, so it runs on its own, now.
          this.startJob(api.apply(this.id, this.project.lang, { requests: [r] }), "Finding speakers again…");
          continue;
        }
        const join = acts.find((a) => a.join != null);
        if (join) {
          const target = this.pending[join.join]?.actions.find((a) => a.type === "new_speaker");
          if (target) {
            target.lines = [...new Set([...target.lines, ...join.lines])];
            this.pending[join.join].request += ` + ${join.lines.length} line(s)`;
            continue;
          }
        }
        const same = (a, b) =>
          a.type === b.type &&
          ((a.type === "recast" && a.speaker === b.speaker) || (["set_text", "set_original"].includes(a.type) && a.line === b.line));
        this.pending = this.pending.filter((p) => !(p.actions.length === 1 && acts.length === 1 && same(p.actions[0], acts[0])));
        this.pending.push({ request: r.request, actions: acts });
      }
      this.savePending();
      this.picked = [];
      toast("Added to your changes — apply them all together when ready.", "ok", 3000);
    },
    removePending(i) {
      this.pending.splice(i, 1);
      this.savePending();
    },
    clearPending() {
      this.pending = [];
      this.savePending();
    },
    async applyAll(lipsync = null) {
      const plan = {
        requests: this.pending.map((r) => ({ request: r.request, actions: r.actions.map(({ join, ...a }) => a) })),
      };
      const n = this.pending.length;
      const label = `Applying ${n} change${n > 1 ? "s" : ""}${lipsync === false ? " (fast, no lip sync)" : ""}…`;
      if (await this.startJob(api.apply(this.id, this.project.lang, plan, lipsync), label)) this.clearPending();
    },
    addLipsync() {
      this.startJob(api.addLipsync(this.id, this.project.lang), "Adding lip sync…");
    },
    fixOriginal(lineId, text) {
      this.queue({ requests: [{ request: `Corrected transcript of line ${lineId}`, actions: [{ type: "set_original", line: lineId, text }] }] });
    },
    saveLine(lineId, text) {
      this.queue({ requests: [{ request: `New wording for line ${lineId}`, actions: [{ type: "set_text", line: lineId, text }] }] });
    },
    revoice(voices) {
      const names = Object.fromEntries((store.options?.voices || []).map((v) => [v.value, v.label.split(" —")[0]]));
      this.queue({
        requests: Object.entries(voices).map(([spk, voice]) => ({
          request: `Speaker ${Number(spk) + 1} → ${names[voice] || voice}`,
          actions: [{ type: "recast", speaker: Number(spk) + 1, voice }],
        })),
      });
    },
    applyPlan(plan) {
      this.queue(plan);
    },
    restored(v) {
      toast(`Restored version ${v}.`);
      this.previewing = null;
      this.load();
    },

    // ── Other languages, sharing ─────────────────────────────────────────
    switchLang(code) {
      go(projectPath(this.id, code));
    },
    addLanguage() {
      const m = this.id.match(/_([0-9.]+)-([0-9.]+|end)$/);
      store.newDub = { url: this.project.url, start: m ? m[1] : "", end: m && m[2] !== "end" ? m[2] : "" };
    },
    dubAgain() {
      this.addLanguage();
      store.newDub.lang = this.project.lang;
    },
    async share() {
      const link = `${location.origin}${location.pathname}#/watch/${encodeURIComponent(this.id)}/${this.project.lang}`;
      try {
        await navigator.clipboard.writeText(link);
        toast("Watch link copied — it opens a clean player with every language.");
      } catch {
        prompt("Copy this watch link:", link);
      }
    },
  },
  template: `
    <main class="page">
      <div v-if="error" class="empty">
        <h2>Could not open this video</h2><p>{{ error }}</p>
        <button class="btn" @click="go('#/')">Back to library</button>
      </div>

      <template v-else-if="project">
        <div class="proj-head">
          <button class="btn ghost icon" title="Back to library" @click="go('#/')">←</button>
          <div class="title-block">
            <h1>{{ project.title }}</h1>
            <div class="sub">{{ project.channel }}<span v-if="project.url && project.url.startsWith('http')"> · <a :href="project.url" target="_blank" rel="noopener" style="text-decoration:underline">Source</a></span></div>
          </div>
          <div class="lang-tabs">
            <button v-for="l in project.langs" :key="l.code" class="lang-tab" :class="{ on: l.code === project.lang }" @click="switchLang(l.code)">{{ l.name }}</button>
            <button class="lang-tab" :disabled="!project.url" title="Dub this video into another language" @click="addLanguage">+ Language</button>
            <button v-if="project.editable && versions.length > 1" class="btn sm" :disabled="busy || undoing"
                    :title="'Undo: ' + versions[0].title" @click="undo">↶ Undo</button>
            <button class="btn sm" title="Copy a link to a clean player for this dub" @click="share">↗ Share</button>
          </div>
        </div>

        <JobProgress v-if="job" :job="job" show-log />

        <div v-if="!project.editable" class="banner">
          <span class="grow">This dub was made before line-by-line data was saved, so it can be watched but not edited. Dubbing it again is quick — finished steps are reused.</span>
          <button class="btn primary sm" :disabled="!project.url" @click="dubAgain">Dub again</button>
        </div>

        <div v-if="project.editable && project.unsynced" class="banner">
          <span class="grow">{{ project.unsynced }} line{{ project.unsynced > 1 ? 's were' : ' was' }} made in fast mode, without lip sync.</span>
          <button class="btn primary sm" :disabled="busy" @click="addLipsync">Add lip sync</button>
        </div>

        <div v-if="guide && project.editable" class="guide">
          <div class="guide-title">Welcome to your studio</div>
          <div class="guide-steps">
            <div><b>1</b><span>▶ <b>Watch</b> your dub. Press <kbd>O</kbd> to hear the original, <kbd>C</kbd> to see both side by side.</span></div>
            <div><b>2</b><span><b>Drag across the film strip</b> to pick a moment you want to change.</span></div>
            <div><b>3</b><span><b>Choose what to change</b> — voice, feeling, words, language — then <b>Apply</b>. ↶ Undo is always there.</span></div>
          </div>
          <button class="btn primary sm" @click="closeGuide">Got it!</button>
        </div>
        <div v-if="confetti.length" class="confetti" aria-hidden="true">
          <i v-for="(c, k) in confetti" :key="k" :style="{ left: c.left + '%', animationDelay: c.delay + 's', background: c.color, '--spin': c.spin + 'deg' }"></i>
        </div>

        <div class="workspace">
          <Player ref="player" :media="media" :lines="project.lines" :speakers="project.speakers"
                  :active-id="activeId" :selected="picked" :langs="project.langs" :lang="project.lang" :changed="lastChanged"
                  :comments="openComments" @comment="openComment"
                  :banner="previewing ? 'Watching version ' + previewing.v + ' — not the current dub' : ''"
                  @time="now = $event" @select="clicked = $event" @range="selectRange" @lang="switchLang" />

          <div class="panel side">
            <EditGuide v-if="project.editable" :step="editStep" :show-tasks="editTips && !pending.length && !pickedLines.length"
                       @go="goEdit" @hide="setEditTips(false)" @show="setEditTips(true)" />
            <PendingBar v-if="pending.length" :requests="pending" :busy="busy" :lipsync="project.lipsync"
                        :project-id="id" :lang="project.lang"
                        @apply="applyAll" @remove="removePending" @clear="clearPending" />
            <SelectionBar v-if="pickedLines.length" :lines="pickedLines" :all-lines="project.lines" :speakers="project.speakers"
                          :new-people="newPeople" :lang="project.lang" :project-id="id" :busy="busy" @apply="applyPlan" @describe="describe" @clear="picked = []" />
            <div class="tabs">
              <button v-for="t in TABS" :key="t[0]" :class="{ on: tab === t[0] }" :title="t[2]" @click="tab = t[0]">{{ t[1] }}</button>
            </div>
            <div class="tab-body">
              <LinesPanel v-show="tab === 'script'" :lines="project.lines" :speakers="project.speakers" :active-id="activeId"
                           :selected="picked" :pending="pendingIds" :editable="project.editable" :busy="busy"
                           @seek="seekTo" @save="saveLine" @fix-original="fixOriginal" @toggle="toggle" @play="playLine" />
              <VoicesPanel v-if="tab === 'cast'" :speakers="project.speakers" :lang="project.lang" :project-id="id"
                         :editable="project.editable" :busy="busy" @revoice="revoice" @plan="applyPlan" />
              <div v-show="tab === 'revise'" class="tab-pad" style="padding-bottom:0">
                <CommentsCard :project-id="id" :lang="project.lang" :comments="comments" :now="now" :span="pickedSpan"
                              :editable="project.editable" @changed="loadComments" @seek="seekTime" @send="sendComment" />
              </div>
              <AskPanel v-show="tab === 'revise'" :project-id="id" :lang="project.lang" :editable="project.editable"
                           :busy="busy" :last-change="lastChange" :scope="pickedLines" :ask="ask" @apply="applyPlan" />
              <VersionsPanel v-if="tab === 'history'" :project-id="id" :lang="project.lang" :editable="project.editable"
                            :busy="busy" :stamp="project.media.dubbed" :previewing="previewing && previewing.v"
                            :lines="project.lines" @preview="preview" @seek="seekTime" @restored="restored" />
              <DownloadPanel v-if="tab === 'export'" :project-id="id" :lang="project.lang" :editable="project.editable" :stamp="project.media.dubbed" />
            </div>
          </div>
        </div>
      </template>

      <div v-else class="hint" style="padding:40px;text-align:center"><span class="spinner"></span></div>
    </main>`,
};
