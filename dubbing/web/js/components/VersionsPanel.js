// Version history of this dub (server/versions.py): every edit, newest
// first, in plain words, with the lines it touched. "Play this change"
// plays that version from its first changed line; "Restore" makes it the
// current dub again (saved as a new version, so that can be undone too).

import { api } from "../api.js";
import { ago, fmtTime, miniMarkdown } from "../util.js";

const ICON = { dub: "film", edit: "pen", revoice: "mic", restore: "loop", lipsync: "lips" };
const KIND = { dub: "Dubbed", edit: "Changed", revoice: "New voices", restore: "Went back", lipsync: "Lip sync added" };

export default {
  props: {
    projectId: { type: String, required: true },
    lang: { type: String, required: true },
    editable: Boolean,
    busy: Boolean,
    stamp: { type: String, default: "" }, // changes when the dub is re-rendered
    previewing: { type: Number, default: null },
    lines: { type: Array, default: () => [] },
  },
  emits: ["preview", "restored", "seek"],
  data: () => ({ versions: [], loading: true, error: "", restoring: null, open: null, ICON, KIND }),
  watch: { lang: "load", stamp: "load" },
  mounted() {
    this.load();
  },
  methods: {
    ago,
    fmtTime,
    miniMarkdown,
    touched(v) {
      const ids = [...(v.changes || "").matchAll(/\*\*S(\d+)\*\*/g)].map((m) => Number(m[1]));
      return this.lines.filter((l) => ids.includes(l.id));
    },
    // Watch this version from its first changed line (the current one just seeks).
    play(v, i) {
      const at = this.touched(v)[0]?.start ?? 0;
      if (i === 0) {
        this.$emit("preview", null);
        this.$emit("seek", at);
      } else this.$emit("preview", { ...v, at });
    },
    async load() {
      this.loading = true;
      try {
        this.versions = await api.versions(this.projectId, this.lang);
        this.error = "";
      } catch (e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },
    async restore(v) {
      this.restoring = v.v;
      try {
        await api.restore(this.projectId, this.lang, v.v);
        this.$emit("restored", v.v);
        await this.load();
      } catch (e) {
        this.error = e.message;
      } finally {
        this.restoring = null;
      }
    },
  },
  template: `
    <div class="tab-pad">
      <div v-if="loading" class="hint">Loading…</div>
      <div v-else-if="error" class="note error">{{ error }}</div>
      <div v-else-if="versions.length < 2" class="empty-mini">
        <div class="big"></div>
        <b>Your changes will show up here</b>
        <p class="hint">Every change is saved as a version. You can play any of them again, or go back to one — nothing is ever lost.</p>
      </div>
      <p v-else class="hint" style="margin:0 0 12px">{{ versions.length - 1 }} change{{ versions.length > 2 ? 's' : '' }} so far. Nothing is ever lost: play any version, or go back to it.</p>
      <div v-for="(v, i) in versions" :key="v.v" class="version" :class="{ current: i === 0, previewing: previewing === v.v }">
        <div class="version-dot"><Icon :name="ICON[v.kind] || 'sparkle'" :size="15" /></div>
        <div class="version-body">
          <div class="version-head">
            <span class="version-title">{{ v.title }}</span>
            <span v-if="i === 0" class="chip accent">now</span>
            <span v-if="previewing === v.v" class="chip">▶ playing</span>
          </div>
          <div class="hint">{{ KIND[v.kind] || 'Version' }} · {{ ago(v.at) }} · v{{ v.v }}</div>
          <div v-if="touched(v).length" class="touched">
            <button v-for="l in touched(v)" :key="l.id" class="chip" :title="l.dubbed" @click="$emit('seek', l.start)">{{ fmtTime(l.start) }} · line {{ l.id }}</button>
          </div>
          <template v-if="v.changes">
            <button class="link" @click="open = open === v.v ? null : v.v">{{ open === v.v ? 'Hide details' : 'What exactly changed?' }}</button>
            <div v-if="open === v.v" class="version-changes" v-html="miniMarkdown(v.changes.replace(/\\*\\*S(\\d+)\\*\\*/g, '**Line $1**'))"></div>
          </template>
          <div class="version-actions">
            <button v-if="previewing !== v.v" class="btn sm" @click="play(v, i)">▶ Play {{ i === 0 ? 'from the change' : 'this version' }}</button>
            <button v-else class="btn sm" @click="$emit('preview', null)">■ Back to now</button>
            <button v-if="i > 0 && editable" class="btn sm" :disabled="busy || restoring !== null" @click="restore(v)">
              <span v-if="restoring === v.v" class="spinner"></span> ↶ Go back to this
            </button>
          </div>
        </div>
      </div>
    </div>`,
};
