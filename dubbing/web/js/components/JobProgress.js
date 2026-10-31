// A running job: time left, its stages as stations on a railway line with
// the train at the current one (util.js STATIONS: a north-to-south line), and any warning about a
// busy GPU. An edit runs fewer stages: an express that stops only where needed.

import { api } from "../api.js";
import { toast } from "../store.js";
import { fmtTime, fmtLeft, finishAt, STAGE_WORDS, STATIONS } from "../util.js";

export default {
  props: { job: { type: Object, required: true }, showLog: { type: Boolean, default: false } },
  data: () => ({ logOpen: false }),
  computed: {
    heading() {
      const j = this.job;
      if (j.status === "queued") return j.position > 1 ? `Waiting — ${j.position - 1} job(s) ahead` : "Starting next…";
      if (j.status === "error") return "Failed";
      if (j.status === "stopped") return "Stopped by you";
      if (j.stopping) return "Stopping after this step…";
      if (j.status === "done") return "Done";
      const stage = j.stages.find((s) => s.key === j.current);
      if (!stage) return "Finishing…";
      if (stage.key === "lipsync" && /off/.test(stage.label)) return "Skipping lip sync (fast mode)…";
      return (STAGE_WORDS[stage.key] || stage.label) + "…";
    },
    verb() {
      return { dub: "Dubbing", revoice: "Re-voicing", edit: "Applying changes", lipsync: "Adding lip sync" }[this.job.kind];
    },
    // The last step writes the dub's files, so it always finishes.
    canStop() {
      const j = this.job;
      if (j.status === "queued") return true;
      return j.status === "running" && !j.stopping && j.current !== j.stages[j.stages.length - 1]?.key;
    },
    // The station the train is at, spelled out under the line on phones.
    nowAt() {
      const j = this.job;
      return j.status === "running" ? j.stages.find((s) => s.key === j.current) : null;
    },
    // Where the train is along the line, in % (stations sit at the middle of
    // equal slices): at the current station, or at the end when done.
    trainAt() {
      const j = this.job;
      const n = j.stages.length;
      const done = j.stages.filter((s) => s.key in j.timings).length;
      const at = j.status === "done" ? n - 1 : Math.min(n - 1, done);
      return j.status === "queued" ? 0 : ((at + 0.5) / n) * 100;
    },
    left() {
      const r = this.job.remaining;
      return r == null ? "" : `Arrives in ${fmtLeft(r).replace("about ", "~")} · around ${finishAt(r)}`;
    },
  },
  methods: {
    async stop() {
      const what = this.job.status === "queued" ? "Cancel this job?" : "Stop after the current step? Finished steps are kept, so starting again is quicker.";
      if (!confirm(what)) return;
      try {
        await api.stop(this.job.id);
      } catch (e) {
        toast(e.message, "error", 6000);
      }
    },
    state(key) {
      if (key in this.job.timings) return "done";
      return key === this.job.current && this.job.status === "running" ? "now" : "";
    },
    tip(s) {
      const took = this.job.timings[s.key];
      if (took != null) return `${s.label}: took ${fmtTime(took)}`;
      const est = this.job.estimates?.[s.key];
      return est != null ? `${s.label}: usually ~${fmtTime(est)}` : s.label;
    },
    station(key) {
      return STATIONS[key] || { code: "" };
    },
    fmtTime,
  },
  template: `
    <div class="jobbar panel">
      <div class="jobbar-top">
        <span v-if="job.status === 'running' || job.status === 'queued'" class="spinner"></span>
        <span class="title">{{ verb }}</span>
        <span class="chip">{{ heading }}</span>
        <span v-if="left" class="eta">{{ left }}</span>
        <span class="elapsed" :title="'Running for ' + fmtTime(job.elapsed)">{{ fmtTime(job.elapsed) }}</span>
        <button v-if="canStop" class="btn sm danger" title="Stop this job — for example to correct something and start again" @click="stop">■ Stop</button>
        <button v-if="showLog" class="btn ghost sm" @click="logOpen = !logOpen">{{ logOpen ? 'Hide' : 'Show' }} details</button>
      </div>
      <div class="rail" :class="{ moving: job.status === 'running' }">
        <div class="track" aria-hidden="true"><i :style="{ width: trainAt + '%' }"></i></div>
        <div class="train" :style="{ left: trainAt + '%' }" aria-hidden="true"><Loco :size="36" /></div>
        <div class="stations">
          <div v-for="s in job.stages" :key="s.key" class="stn" :class="state(s.key)" :title="tip(s)">
            <i class="stop"></i>
            <span class="stn-en">{{ s.label }}</span>
            <span class="stn-code" :title="station(s.key).place">{{ station(s.key).code }}</span>
          </div>
        </div>
        <div v-if="nowAt" class="rail-now">Now: <b>{{ nowAt.label }}</b> · near {{ station(nowAt.key).place }}</div>
      </div>
      <div v-for="n in job.notes || []" :key="n" class="note warn" style="margin: 12px 0 0">{{ n }}</div>
      <div v-if="job.status === 'error'" class="note error" style="margin: 12px 0 0">{{ job.message }}</div>
      <div v-if="job.status === 'stopped'" class="note" style="margin: 12px 0 0">{{ job.message }}</div>
      <div v-if="logOpen" class="log">{{ job.log.join('\\n') || 'Nothing logged yet.' }}</div>
    </div>`,
};
