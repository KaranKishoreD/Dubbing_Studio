// Review comments (server/comments.py): notes left on a moment of the dub,
// by name — here or from a watch link. Each can jump to its moment, be sent
// to the Ask tab as an editable request, or be marked done.

import { api } from "../api.js";
import { toast } from "../store.js";
import { ago, fmtTime } from "../util.js";

export const commenter = {
  get() {
    try {
      return localStorage.getItem("commenter") || "";
    } catch {
      return "";
    }
  },
  set(name) {
    try {
      localStorage.setItem("commenter", name);
    } catch {
      /* asked again next time */
    }
  },
};

export default {
  props: {
    projectId: { type: String, required: true },
    lang: { type: String, required: true },
    comments: { type: Array, default: () => [] },
    now: { type: Number, default: 0 }, // the player's time: where a new comment goes
    span: { type: Array, default: null }, // [a, b] of the marked moment, if any
    editable: Boolean,
  },
  emits: ["changed", "seek", "send"],
  data: () => ({ name: commenter.get(), text: "", showDone: false, adding: false }),
  computed: {
    open() {
      return this.comments.filter((c) => !c.done).sort((a, b) => a.t - b.t);
    },
    done() {
      return this.comments.filter((c) => c.done);
    },
    where() {
      return this.span ? `${fmtTime(this.span[0])}–${fmtTime(this.span[1])}` : fmtTime(this.now);
    },
  },
  methods: {
    ago,
    fmtTime,
    async add() {
      if (!this.name.trim() || !this.text.trim()) return;
      commenter.set(this.name.trim());
      const [t, t2] = this.span || [this.now, null];
      try {
        await api.addComment(this.projectId, { lang: this.lang, name: this.name, text: this.text, t, t2 });
        this.text = "";
        this.adding = false;
        this.$emit("changed");
      } catch (e) {
        toast(e.message, "error", 5000);
      }
    },
    async mark(c, done) {
      await api.updateComment(this.projectId, c.id, { lang: this.lang, done });
      this.$emit("changed");
    },
  },
  template: `
    <div class="comments">
      <div class="cm-head">
        <span><b>Comments</b> <span class="hint" v-if="open.length">— {{ open.length }} open</span></span>
        <button class="btn ghost sm" @click="adding = !adding">{{ adding ? 'Cancel' : '+ Comment at ' + where }}</button>
      </div>
      <div v-if="adding" class="cm-form">
        <input class="input sm" v-model="name" placeholder="Your name" style="max-width:150px" />
        <input class="input sm" v-model="text" :placeholder="'What should change at ' + where + '?'" @keydown.enter="add" />
        <button class="btn sm primary" :disabled="!name.trim() || !text.trim()" @click="add">Post</button>
      </div>
      <div v-if="!open.length && !adding" class="hint" style="padding:2px 0 4px">No open comments. Share the watch link (↗ Share) so others can leave some.</div>
      <div v-for="c in open" :key="c.id" class="cm">
        <button class="chip cm-time" :title="'Jump to ' + fmtTime(c.t)" @click="$emit('seek', c.t)">{{ fmtTime(c.t) }}{{ c.t2 ? '–' + fmtTime(c.t2) : '' }}</button>
        <div class="cm-body">
          <div><b>{{ c.name }}</b> <span class="hint">· {{ ago(c.at) }}</span></div>
          <div>{{ c.text }}</div>
          <div class="cm-actions">
            <button v-if="editable" class="btn sm" title="Put this comment in the Ask box to edit and plan" @click="$emit('send', c)">Send to Ask</button>
            <button class="btn ghost sm" @click="mark(c, true)">✓ Done</button>
          </div>
        </div>
      </div>
      <button v-if="done.length" class="link" @click="showDone = !showDone">{{ showDone ? 'Hide' : 'Show' }} {{ done.length }} done</button>
      <div v-if="showDone" v-for="c in done" :key="c.id" class="cm done">
        <span class="chip cm-time">{{ fmtTime(c.t) }}</span>
        <div class="cm-body"><b>{{ c.name }}</b>: {{ c.text }} <button class="link" @click="mark(c, false)">reopen</button></div>
      </div>
    </div>`,
};
