// Names & words (pipeline/text/glossary.py): terms the dub must get right.
// Saving the list does not redo anything; the card then offers to re-translate
// just the lines that use the words, as a waiting change.

import { api } from "../api.js";
import { toast } from "../store.js";

export default {
  props: {
    projectId: { type: String, required: true },
    lang: { type: String, required: true },
    editable: Boolean,
  },
  emits: ["apply"],
  data: () => ({ open: false, rows: [], lines: {}, saved: "", saving: false }),
  computed: {
    dirty() {
      return JSON.stringify(this.clean()) !== this.saved;
    },
    using() {
      return [...new Set(Object.values(this.lines).flat())].sort((a, b) => a - b);
    },
  },
  watch: { lang: "load" },
  mounted() {
    this.load();
  },
  methods: {
    clean() {
      return this.rows.filter((r) => r.term.trim()).map((r) => ({ term: r.term.trim(), say: r.say.trim() }));
    },
    async load() {
      try {
        const g = await api.glossary(this.projectId, this.lang);
        this.rows = g.entries.map((r) => ({ ...r }));
        this.lines = g.lines;
        this.saved = JSON.stringify(this.clean());
      } catch {
        this.rows = [];
      }
      if (!this.rows.length) this.rows.push({ term: "", say: "" });
    },
    add() {
      this.rows.push({ term: "", say: "" });
      this.$nextTick(() => [...this.$el.querySelectorAll(".gl-term")].pop()?.focus());
    },
    async save() {
      this.saving = true;
      try {
        const g = await api.saveGlossary(this.projectId, this.lang, this.clean());
        this.lines = g.lines;
        this.saved = JSON.stringify(this.clean());
        toast("Names & words saved — every new translation follows them.");
      } catch (e) {
        toast(e.message, "error", 6000);
      } finally {
        this.saving = false;
      }
    },
    redoLines() {
      const ids = this.using;
      this.$emit("apply", {
        requests: [
          {
            request: `Use the names & words list on ${ids.length} line(s)`,
            actions: [{ type: "retranslate", lines: ids, instruction: "Follow the names & words list exactly." }],
          },
        ],
      });
    },
  },
  template: `
    <div class="glossary">
      <button class="gl-head" @click="open = !open">
        <span><b>Names & words</b> <span class="hint">— keep a name, a brand or a term exactly right</span></span>
        <span class="chip" v-if="clean().length">{{ clean().length }}</span>
        <span>{{ open ? '▴' : '▾' }}</span>
      </button>
      <div v-if="open" class="gl-body">
        <div class="hint" style="margin-bottom:8px">Leave “say it as” empty to keep the word as it is (e.g. <i>Muthuvel</i>, <i>convex set</i>). Fill it in to say it your way (e.g. Rajini → रजनी).</div>
        <div v-for="(r, i) in rows" :key="i" class="gl-row">
          <input class="input sm gl-term" v-model="r.term" placeholder="Word in the video" :disabled="!editable" />
          <span class="hint">→</span>
          <input class="input sm" v-model="r.say" placeholder="say it as (optional)" :disabled="!editable" />
          <span v-if="lines[r.term] && lines[r.term].length" class="chip" :title="'Lines ' + lines[r.term].join(', ')">{{ lines[r.term].length }} line{{ lines[r.term].length > 1 ? 's' : '' }}</span>
          <button class="btn ghost icon sm" title="Remove" :disabled="!editable" @click="rows.splice(i, 1)">✕</button>
        </div>
        <div class="gl-actions">
          <button class="btn ghost sm" :disabled="!editable" @click="add">+ Add a word</button>
          <span class="spacer"></span>
          <button class="btn sm primary" :disabled="!editable || !dirty || saving" @click="save">Save list</button>
        </div>
        <div v-if="!dirty && using.length" class="note" style="margin-top:10px">
          {{ using.length }} line{{ using.length > 1 ? 's use' : ' uses' }} these words.
          <button class="btn sm" style="margin-left:8px" @click="redoLines">Re-translate just those</button>
        </div>
      </div>
    </div>`,
};
