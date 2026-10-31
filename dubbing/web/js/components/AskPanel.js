// Ask for changes in plain words. The request is first turned into a list of
// concrete changes to review; "Apply" then redoes only the affected lines.

import { api } from "../api.js";
import { fmtTime, miniMarkdown } from "../util.js";
import GlossaryCard from "./GlossaryCard.js";

const EXAMPLES = [
  "Make the first line more casual",
  "Speaker 2 is an older man with a deeper voice",
  "Line 5 should sound sad",
  "Lines 4 to 6 are a different woman",
  "Speaker 3 is the same person as Speaker 1",
  "Don't dub the song",
];

export default {
  components: { GlossaryCard },
  props: {
    projectId: { type: String, required: true },
    lang: { type: String, required: true },
    editable: Boolean,
    busy: Boolean,
    lastChange: { type: String, default: "" }, // what the last applied edit changed
    scope: { type: Array, default: () => [] }, // selected lines: the request applies to these only
    ask: { type: Object, default: null }, // {text}: plan this right away (selection bar); {text, edit}: just fill the box (a comment)
  },
  emits: ["apply"],
  data: () => ({ message: "", planning: false, result: null, error: "", EXAMPLES }),
  watch: {
    ask(a) {
      if (!a?.text) return;
      this.message = a.text;
      this.result = null;
      if (a.edit) this.$nextTick(() => this.$refs.box?.focus());
      else this.plan();
    },
  },
  computed: {
    canApply() {
      return this.result?.plan?.requests?.some((r) => r.actions?.length);
    },
    scopeText() {
      if (!this.scope.length) return "";
      const a = Math.min(...this.scope.map((l) => l.start));
      const b = Math.max(...this.scope.map((l) => l.end));
      const ids = this.scope.map((l) => l.id);
      return `${ids.length === 1 ? "line" : "lines"} ${ids.join(", ")} (${fmtTime(a)}–${fmtTime(b)})`;
    },
  },
  methods: {
    miniMarkdown,
    async plan() {
      this.error = "";
      this.result = null;
      this.planning = true;
      try {
        // With lines selected, the planner is told to touch only those.
        const msg = this.scope.length ? `For ${this.scopeText} only: ${this.message}` : this.message;
        this.result = await api.plan(this.projectId, this.lang, msg);
      } catch (e) {
        this.error = e.message;
      } finally {
        this.planning = false;
      }
    },
    apply() {
      this.$emit("apply", this.result.plan);
      this.result = null;
      this.message = "";
    },
  },
  template: `
    <div class="tab-pad">
      <GlossaryCard :project-id="projectId" :lang="lang" :editable="editable" @apply="$emit('apply', $event)" />
      <div v-if="scope.length" class="scope">Applies to {{ scopeText }} <span class="hint">— clear the selection to change anything</span></div>
      <div class="field">
        <label>What should change?</label>
        <textarea ref="box" class="input" rows="4" v-model="message" :disabled="!editable"
          placeholder="Any number of changes in one message — wording, tone, voices, who said a line, leaving a line undubbed…"
          @keydown.ctrl.enter="plan" @keydown.meta.enter="plan"></textarea>
      </div>
      <div class="examples">
        <button v-for="e in EXAMPLES" :key="e" @click="message = message ? message + ', ' + e.toLowerCase() : e">{{ e }}</button>
      </div>
      <button class="btn" style="width:100%" :disabled="!editable || planning || !message.trim()" @click="plan">
        <span v-if="planning" class="spinner"></span> {{ planning ? 'Reading your request…' : 'Plan changes' }}
      </button>

      <div v-if="error" class="note error" style="margin-top:14px">{{ error }}</div>

      <div v-if="result" style="margin-top:16px">
        <div class="note" :class="{ warn: result.status.includes('Not handled') }" v-html="miniMarkdown(result.status)"></div>
        <div v-for="(c, i) in result.changes" :key="i" class="change">
          <div class="req">“{{ c.request }}”</div>
          <div class="what">{{ c.change }} <span class="chip">{{ c.target.replace(/S(\\d+)/g, 'line $1') }}</span></div>
          <div v-if="c.detail" class="detail">{{ c.detail }}</div>
        </div>
        <button class="btn primary" style="width:100%;margin-top:6px" :disabled="!canApply || busy" @click="apply">Add to changes</button>
      </div>

      <div v-if="lastChange" style="margin-top:18px">
        <div class="label" style="margin-bottom:8px">What changed last time</div>
        <div class="note" v-html="miniMarkdown(lastChange.replace(/\\*\\*S(\\d+)\\*\\*/g, '**Line $1**'))"></div>
      </div>
    </div>`,
};
