// A new dub in progress. Lines appear as soon as they exist: first the
// transcript, then the translation, then each line's new voice to play.
// When the dub finishes, the page opens the result.

import { api } from "../api.js";
import { store, track, go, projectPath } from "../store.js";
import { fmtTime, speakerColor, speakerName, youtubeId } from "../util.js";
import JobProgress from "../components/JobProgress.js";

export default {
  components: { JobProgress },
  props: { id: { type: Number, required: true } },
  data: () => ({ store, error: "", lines: [], timer: 0, playing: null, audio: null, tip: 0, tipTimer: 0 }),
  computed: {
    job() {
      return store.jobs[this.id];
    },
    thumb() {
      const v = youtubeId(this.job?.title);
      return v ? `https://i.ytimg.com/vi/${v}/mqdefault.jpg` : "";
    },
    tips: () => [
      "When it's ready, press O to flip between the original and the dub.",
      "Drag across the film strip to pick any moment and change it — voice, feeling, words or language.",
      "Press C for Compare: the original and the dub side by side, so you can see the lips move.",
      "Every change is saved as a version. Nothing is ever lost — ↶ Undo is always there.",
      "In a hurry? Untick Lip sync next time for a much faster draft, and add lip sync later.",
      "You can leave this page. The dub keeps going and appears on your home page.",
    ],
    phase() {
      const voiced = this.lines.filter((l) => l.audio).length;
      if (voiced) return `${voiced} of ${this.lines.length} lines voiced — press ▶ to listen`;
      if (this.lines.some((l) => l.dubbed)) return "Translated — voices come next";
      return this.lines.length ? "Transcript — translation comes next" : "";
    },
  },
  watch: {
    "job.status"(status) {
      if (status === "done") {
        store.celebrate = true; // the studio greets the new dub
        go(projectPath(this.job.project, this.job.lang));
      }
    },
  },
  async created() {
    if (!this.job) {
      try {
        const j = await api.job(this.id);
        if (j.status === "done") return go(projectPath(j.project, j.lang));
        track(j);
      } catch (e) {
        this.error = e.message;
        return;
      }
    }
    this.poll();
    this.tipTimer = setInterval(() => (this.tip = (this.tip + 1) % this.tips.length), 7000);
  },
  unmounted() {
    clearInterval(this.tipTimer);
    clearTimeout(this.timer);
    this.audio?.pause();
  },
  methods: {
    fmtTime,
    speakerColor,
    speakerName,
    async poll() {
      try {
        this.lines = await api.jobLines(this.id);
      } catch {
        /* keep the last lines */
      }
      if (this.job && (this.job.status === "running" || this.job.status === "queued")) {
        this.timer = setTimeout(this.poll, 3000);
      }
    },
    play(l) {
      this.audio?.pause();
      if (this.playing === l.id) return (this.playing = null);
      this.playing = l.id;
      this.audio = new Audio(l.audio);
      this.audio.onended = this.audio.onerror = () => (this.playing = null);
      this.audio.play().catch(() => (this.playing = null));
    },
  },
  template: `
    <main class="page">
      <div class="job-page">
        <div v-if="error" class="empty"><h2>Job not found</h2><p>{{ error }}</p><button class="btn" @click="go('#/')">Back to library</button></div>
        <template v-else-if="job">
          <div class="hero">
            <img v-if="thumb" :src="thumb" alt="" />
            <div>
              <h1 style="margin:0 0 6px;font-size:22px">{{ { error: 'The dub failed', stopped: 'You stopped this dub' }[job.status] || 'Making your dub' }}</h1>
              <div class="hint">{{ job.title }}</div>
              <div v-if="job.status === 'running' || job.status === 'queued'" class="hint" style="margin-top:6px">You can leave this page — the dub keeps going and appears in your library.</div>
            </div>
          </div>
          <JobProgress :job="job" show-log />
          <div v-if="job.status === 'running' || job.status === 'queued'" class="tip-card" :key="tip">{{ tips[tip] }}</div>

          <div v-if="lines.length" class="panel live">
            <div class="live-head"><b>Lines so far</b><span class="hint">{{ phase }}</span></div>
            <div v-for="l in lines" :key="l.id" class="live-line">
              <span class="line-time">{{ fmtTime(l.start) }}</span>
              <span class="dot" :style="{ background: speakerColor(l.speaker) }" :title="speakerName(l.speaker)"></span>
              <div class="live-text">
                <div class="line-orig">{{ l.text }}</div>
                <div v-if="l.dubbed" class="line-dub">{{ l.dubbed }}</div>
                <div v-else-if="!l.kept" class="hint">translating…</div>
              </div>
              <button v-if="l.audio" class="btn icon sm" :title="'Listen to line ' + l.id" @click="play(l)">
                {{ playing === l.id ? '■' : '▶' }}
              </button>
            </div>
          </div>

          <div style="display:flex;gap:10px;margin-top:16px">
            <button v-if="job.status === 'stopped' && job.request" class="btn primary" @click="store.newDub = { ...job.request }">Change settings &amp; start again</button>
            <button class="btn" @click="go('#/')">Back to library</button>
          </div>
        </template>
        <div v-else class="hint" style="text-align:center"><span class="spinner"></span></div>
      </div>
    </main>`,
};
