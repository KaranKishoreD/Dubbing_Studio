// Changes waiting to be applied. Every button in the studio (selection bar,
// voices, line edits, the Ask tab) adds to this list instead of starting work;
// "Apply all" then runs one job that redoes only the affected lines; before
// that, the bar says which lines that is and about how long it will take.

import { api } from "../api.js";
import { fmtLeft } from "../util.js";

export default {
  props: {
    requests: { type: Array, required: true }, // [{request, actions}]
    projectId: { type: String, required: true },
    lang: { type: String, required: true },
    busy: Boolean,
    lipsync: { type: Boolean, default: true }, // this dub's own setting
  },
  emits: ["apply", "remove", "clear"],
  data() {
    return { sync: this.lipsync, impact: null, timer: 0 };
  },
  computed: {
    impactText() {
      const i = this.impact;
      if (!i) return "";
      const n = i.lines.length;
      const which = n <= 6 ? (n === 1 ? "line " : "lines ") + i.lines.join(", ") : `${n} lines`;
      return `Will redo ${which} · ${fmtLeft(i.seconds)}`;
    },
  },
  watch: {
    requests: { deep: true, immediate: true, handler: "measure" },
    sync: "measure",
  },
  unmounted() {
    clearTimeout(this.timer);
  },
  methods: {
    // Ask the server what applying would redo (debounced; nothing runs).
    measure() {
      clearTimeout(this.timer);
      this.timer = setTimeout(async () => {
        const plan = { requests: this.requests.map((r) => ({ request: r.request, actions: r.actions.map(({ join, ...a }) => a) })) };
        try {
          this.impact = await api.impact(this.projectId, this.lang, plan, this.sync);
        } catch {
          this.impact = null;
        }
      }, 350);
    },
  },
  template: `
    <div class="pending-bar">
      <div class="pending-head">
        <b>{{ requests.length }} change{{ requests.length > 1 ? 's' : '' }} waiting</b>
        <span class="hint">nothing is redone until you apply</span>
        <span v-if="impact" class="impact" :title="'Only these lines are redone; the rest of the dub stays exactly as it is'">{{ impactText }}</span>
      </div>
      <ol class="pending-list">
        <li v-for="(r, i) in requests" :key="i">
          <span>{{ r.request }}</span>
          <button class="btn ghost icon sm" title="Remove this change" @click="$emit('remove', i)">✕</button>
        </li>
      </ol>
      <div class="pending-actions">
        <label class="switch-row sm" title="Off = fast: changed lines are not lip-synced now (add it later)">
          <input type="checkbox" v-model="sync" /> Lip sync
        </label>
        <button class="btn ghost sm" @click="$emit('clear')">Discard all</button>
        <button class="btn primary sm" :disabled="busy" @click="$emit('apply', sync)">Apply all &amp; re-dub</button>
      </div>
    </div>`,
};
