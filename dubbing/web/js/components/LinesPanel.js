// Every line: who says it, the original and the dub.
//   click a line     jump the video there
//   ▶ Orig / ▶ Dub   hear just that line, in the original or the dub
//   ☐                select it (shift-click selects a run of lines); the
//                    page then offers changes for the selected lines
//   ✎ Edit           type new dubbed wording; only that line is re-voiced

import { store } from "../store.js";
import { fmtTime, speakerName } from "../util.js";
import Avatar from "./Avatar.js";

export default {
  components: { Avatar },
  props: {
    lines: { type: Array, required: true },
    speakers: { type: Array, default: () => [] },
    activeId: { type: Number, default: null },
    selected: { type: Array, default: () => [] },
    pending: { type: Array, default: () => [] }, // lines a waiting change will redo
    editable: Boolean,
    busy: Boolean,
  },
  emits: ["seek", "save", "fix-original", "toggle", "play"],
  // editing: the line being edited; field: "dubbed" (the dub's wording) or
  // "text" (the original transcript, which is then translated again).
  data: () => ({ follow: true, editing: null, field: "dubbed", draft: "", query: "" }),
  computed: {
    shown() {
      const q = this.query.trim().toLowerCase();
      if (!q) return this.lines;
      return this.lines.filter((l) => (l.text + " " + l.dubbed).toLowerCase().includes(q));
    },
    speakerById() {
      return Object.fromEntries(this.speakers.map((s) => [s.id, s]));
    },
  },
  watch: {
    activeId(id) {
      if (!this.follow || this.editing !== null) return;
      this.$nextTick(() =>
        this.$el.querySelector(`[data-line="${id}"]`)?.scrollIntoView({ block: "nearest", behavior: "smooth" }),
      );
    },
  },
  methods: {
    fmtTime,
    speakerName,
    langName(code) {
      return (store.options?.languages || []).find((l) => l.code === code)?.label.split(" (")[0] || code;
    },
    edit(line, field = "dubbed") {
      this.editing = line.id;
      this.field = field;
      this.draft = line[field];
      this.$nextTick(() => this.$el.querySelector(".line-edit textarea")?.focus());
    },
    save(line) {
      const text = this.draft.trim();
      if (text && text !== line[this.field]) this.$emit(this.field === "text" ? "fix-original" : "save", line.id, text);
      this.editing = null;
    },
  },
  template: `
    <div>
      <div class="follow">
        <label><input type="checkbox" v-model="follow" /> Follow playback</label>
        <input class="input" style="height:28px;width:160px;font-size:12px" v-model="query" placeholder="Search lines" />
      </div>
      <div v-if="!lines.length" class="tab-pad hint">No lines saved for this dub.</div>
      <div v-for="l in shown" :key="l.id" :data-line="l.id" class="line"
           :class="{ active: l.id === activeId, picked: selected.includes(l.id), pending: pending.includes(l.id) }"
           @click="editing !== l.id && $emit('seek', l)">
        <div class="line-side">
          <input v-if="editable" type="checkbox" class="pick" :checked="selected.includes(l.id)" title="Select (shift-click for a run of lines)"
                 @click.stop="$emit('toggle', l.id, $event.shiftKey)" />
          <div class="line-time">{{ fmtTime(l.start) }}</div>
        </div>
        <div>
          <div class="line-who">
            <Avatar v-if="speakerById[l.speaker]" :speaker="speakerById[l.speaker]" :size="18" />
            <span>{{ speakerName(l.speaker) }}</span>
            <span v-if="l.voice" style="color:var(--text-faint)">· {{ l.voice }}</span>
            <span v-if="l.sung" class="chip">♪ sung</span>
            <span v-if="l.lang" class="chip" title="This line is dubbed in another language">{{ langName(l.lang) }}</span>
            <span v-if="pending.includes(l.id)" class="chip accent" title="A waiting change will redo this line">● changing</span>
            <span v-if="l.flag === 'low'" class="chip warn" title="The transcript may be wrong here — worth a listen">check</span>
            <span v-if="l.flag === 'gap'" class="chip" title="Recovered from speech the recogniser first skipped">＋ recovered</span>
          </div>
          <div v-if="!(editing === l.id && field === 'text')" class="line-orig">{{ l.text }}</div>
          <div v-if="editing !== l.id || field === 'text'" class="line-dub" :class="{ kept: l.kept }">{{ l.kept ? 'Original audio kept' : l.dubbed }}</div>
          <div v-if="editing === l.id" class="line-edit" @click.stop>
            <div class="hint" style="margin-bottom:4px">{{ field === 'text' ? 'Correct what was said — the line is then translated again' : 'New wording for the dub' }}</div>
            <textarea class="input" rows="3" v-model="draft" @keydown.enter.exact.prevent="save(l)" @keydown.esc="editing = null"></textarea>
            <div class="row">
              <span class="hint" style="margin-right:auto">Enter to save · Esc to cancel</span>
              <button class="btn sm" @click="editing = null">Cancel</button>
              <button class="btn sm primary" @click="save(l)">Add to changes</button>
            </div>
          </div>
        </div>
        <div v-if="editing !== l.id" class="line-actions" @click.stop>
          <button class="btn sm" title="Hear the original line" @click="$emit('play', l, 'orig')">▶ Orig</button>
          <button v-if="!l.kept" class="btn sm" title="Hear the dubbed line" @click="$emit('play', l, 'dub')">▶ Dub</button>
          <button v-if="editable && !l.kept" class="btn sm" title="Change the dubbed wording" @click="edit(l)">✎ Dub</button>
          <button v-if="editable" class="btn sm" title="Correct the original transcript (wrong or unwanted words); the line is translated again" @click="edit(l, 'text')">✎ Transcript</button>
        </div>
      </div>
    </div>`,
};
