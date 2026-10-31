// "What should change here?" — shown when lines are selected (a moment
// marked on the timeline, or lines ticked in the script). Big tiles for the
// common changes; each opens its choices right below. Nothing is redone
// straight away: every choice joins the list of waiting changes.
//
// Each choice builds the same kind of plan the Ask planner returns
// (pipeline/edits/edit_planner.py), and the server checks it the same way.

import { api } from "../api.js";
import { store, toast } from "../store.js";
import { fmtTime, speakerName, speakerColor, splitLangLabel } from "../util.js";

// Friendly names for the speech service's delivery styles.
const FEELINGS = {
  happy: "Happy / excited",
  sad: "Sad",
  anger: "Angry",
  surprise: "Surprised",
  fear: "Scared",
  disgust: "Disgusted",
  "Customer Care": "Warm & friendly",
  advertisements: "Like an advert",
  "single person narration audiobook": "Storytelling",
  "children's stories": "Children's story",
  "educational lecture": "Teacher",
  news: "News",
  "TV style news": "TV news",
  "AIR style news": "Radio news",
};

const TILES = [
  ["voice", "mic", "Voice", "Give these lines another voice"],
  ["speaker", "user", "Who's speaking", "Wrong person? Fix it"],
  ["feeling", "smile", "Feeling", "Happier, sadder, calmer…"],
  ["language", "globe", "Language", "Say these lines in another language"],
  ["keep", "pause", "Keep original", "Leave this part undubbed"],
  ["lips", "lips", "Lip sync", "Render the mouths again — the audio stays as it is"],
  ["words", "pen", "In your words", "Describe any other change"],
];

export default {
  props: {
    lines: { type: Array, required: true }, // the selected lines
    allLines: { type: Array, required: true },
    speakers: { type: Array, required: true },
    newPeople: { type: Array, default: () => [] }, // new people in waiting changes
    projectId: { type: String, required: true },
    lang: { type: String, default: "hi" },
    busy: Boolean,
  },
  emits: ["apply", "describe", "clear"],
  data: () => ({ open: null, text: "", playing: null, trying: null, audio: null, TILES, FEELINGS }),
  computed: {
    ids() {
      return this.lines.map((l) => l.id);
    },
    span() {
      const a = Math.min(...this.lines.map((l) => l.start));
      const b = Math.max(...this.lines.map((l) => l.end));
      return `${fmtTime(a)} – ${fmtTime(b)}`;
    },
    who() {
      return [...new Set(this.lines.map((l) => l.speaker))];
    },
    // If every line of one speaker is selected, a voice change is a change
    // to that speaker; otherwise these lines become a speaker of their own.
    wholeSpeaker() {
      if (this.who.length !== 1) return null;
      const s = this.who[0];
      return this.allLines.filter((l) => l.speaker === s).length === this.lines.length ? s : null;
    },
    voices: () => store.options?.voices || [],
    styles: () => store.options?.delivery_styles || [],
    languages: () => (store.options?.languages || []).map((l) => ({ code: l.code, ...splitLangLabel(l.label) })),
    // The line a sample is made of: the longest selected one.
    sample() {
      return [...this.lines].sort((a, b) => b.end - b.start - (a.end - a.start))[0];
    },
    allKept() {
      return this.lines.every((l) => l.kept);
    },
  },
  watch: {
    lines() {
      this.audio?.pause();
    },
  },
  unmounted() {
    this.audio?.pause();
  },
  methods: {
    speakerName,
    speakerColor,
    feeling(s) {
      return FEELINGS[s] || s;
    },
    toggle(key) {
      if (key === "keep") return this.keep();
      if (key === "lips") return this.plan(`Lip sync ${this.ids.length} line(s) again`, [{ type: "lip_sync", lines: this.ids }]);
      this.open = this.open === key ? null : key;
      if (key === "words") this.$nextTick(() => this.$el.querySelector(".words input")?.focus());
    },
    plan(label, actions) {
      this.$emit("apply", { requests: [{ request: label, actions }] }, label);
      this.open = null;
    },
    // ▶ Try: hear the longest selected line made exactly as this choice
    // would make it (one speech call; replaying is free).
    async tryIt(key, action) {
      this.audio?.pause();
      if (this.playing === key) return (this.playing = null);
      this.trying = key;
      try {
        const { audio } = await api.audition(this.projectId, this.lang, this.sample.id, action);
        this.trying = null;
        this.playing = key;
        this.audio = new Audio(audio);
        this.audio.onended = this.audio.onerror = () => (this.playing = null);
        this.audio.play().catch(() => (this.playing = null));
      } catch (e) {
        this.trying = null;
        toast(e.message, "error", 6000);
      }
    },
    icon(key) {
      return this.trying === key ? "" : this.playing === key ? "■" : "▶";
    },
    voiceAction(v) {
      return this.wholeSpeaker != null
        ? { type: "recast", speaker: this.wholeSpeaker + 1, voice: v.value }
        : { type: "new_speaker", lines: [this.sample.id], gender: v.gender, voice: v.value };
    },
    setVoice(v) {
      if (v === "steady") {
        return this.plan(`Steady voice for ${this.ids.length} line(s)`, [{ type: "steady_voice", lines: this.ids }]);
      }
      const name = v.label.split(" —")[0];
      if (this.wholeSpeaker != null) {
        this.plan(`${speakerName(this.wholeSpeaker)} → ${name}`, [{ type: "recast", speaker: this.wholeSpeaker + 1, voice: v.value }]);
      } else {
        this.plan(`${this.ids.length} line(s) → voice ${name}`, [
          { type: "new_speaker", lines: this.ids, gender: v.gender, voice: v.value },
        ]);
      }
    },
    setSpeaker(val) {
      if (typeof val === "object") {
        this.plan(`${this.ids.length} line(s) → ${val.label}`, [
          { type: "new_speaker", lines: this.ids, gender: val.gender, join: val.idx },
        ]);
      } else if (val === "male" || val === "female") {
        this.plan(`${this.ids.length} line(s) → a new ${val === "male" ? "man" : "woman"}`, [
          { type: "new_speaker", lines: this.ids, gender: val },
        ]);
      } else {
        this.plan(`${this.ids.length} line(s) → ${speakerName(val)}`, [
          { type: "reassign_speaker", lines: this.ids, speaker: val + 1 },
        ]);
      }
    },
    setStyle(style) {
      this.plan(`${this.feeling(style)} for ${this.ids.length} line(s)`, [{ type: "set_style", lines: this.ids, style }]);
    },
    setLang(l) {
      this.plan(`${this.ids.length} line(s) in ${l.en}`, [{ type: "set_language", lines: this.ids, lang: l.code }]);
    },
    keep() {
      this.plan(this.allKept ? "Dub again" : "Keep the original audio", [
        { type: this.allKept ? "dub_again" : "keep_original", lines: this.ids },
      ]);
    },
    describe(text) {
      const t = (text ?? this.text).trim();
      if (!t) return;
      this.$emit("describe", t);
      this.text = "";
      this.open = null;
    },
  },
  template: `
    <div class="selcard">
      <div class="selcard-head">
        <div>
          <div class="selcard-q">What should change here?</div>
          <div class="hint">{{ span }} · {{ lines.length }} line{{ lines.length > 1 ? 's' : '' }} ·
            <span v-for="s in who" :key="s" class="who"><i :style="{ background: speakerColor(s) }"></i>{{ speakerName(s) }}</span></div>
        </div>
        <button class="btn ghost icon sm" title="Clear the selection" @click="$emit('clear')">✕</button>
      </div>

      <div class="tiles">
        <button v-for="t in TILES" :key="t[0]" class="tile" :class="{ on: open === t[0] }" :title="t[3]" @click="toggle(t[0])">
          <span class="tile-icon"><Icon :name="t[1]" :size="22" /></span>
          <span class="tile-name">{{ t[0] === 'keep' && allKept ? 'Dub again' : t[2] }}</span>
        </button>
      </div>

      <div v-if="open === 'voice'" class="choices">
        <div class="hint">▶ plays <b>“{{ (sample.dubbed || sample.text).slice(0, 60) }}{{ (sample.dubbed || sample.text).length > 60 ? '…' : '' }}”</b> in that voice — then click the name to choose it.</div>
        <span class="voice-chip wide">
          <button class="play" title="Hear this line" @click="tryIt('steady', { type: 'steady_voice', lines: [sample.id] })">{{ icon('steady') }}</button>
          <button class="pick" @click="setVoice('steady')">Their own voice, steady <span class="hint">— fixes a voice that wobbles or sounds like someone else</span></button>
        </span>
        <div v-for="g in ['female', 'male']" :key="g" class="choice-group">
          <div class="label">{{ g === 'female' ? 'Women' : 'Men' }}</div>
          <div class="choice-row">
            <span v-for="v in voices.filter(x => x.gender === g)" :key="v.value" class="voice-chip">
              <button class="play" :title="'Hear this line in ' + v.label.split(' —')[0] + ' voice'" @click="tryIt(v.value, voiceAction(v))">{{ icon(v.value) }}</button>
              <button class="pick" :title="v.label" @click="setVoice(v)">{{ v.label.split(' —')[0] }}<small v-if="v.value.startsWith('sarvam:')" class="hint"> · Sarvam</small></button>
            </span>
          </div>
        </div>
      </div>

      <div v-if="open === 'speaker'" class="choices">
        <div class="choice-row">
          <button v-for="s in speakers" :key="s.id" class="choice" @click="setSpeaker(s.id)">
            <i class="dot" :style="{ background: speakerColor(s.id) }"></i>{{ speakerName(s.id) }}</button>
          <button v-for="p in newPeople" :key="'n' + p.idx" class="choice" @click="setSpeaker(p)">{{ p.label }}</button>
          <button class="choice" @click="setSpeaker('male')">+ A new man</button>
          <button class="choice" @click="setSpeaker('female')">+ A new woman</button>
        </div>
      </div>

      <div v-if="open === 'feeling'" class="choices">
        <div class="label">Speed</div>
        <div class="choice-row">
          <span v-for="p in [['fast', 'Faster'], ['normal', 'Normal speed'], ['slow', 'Slower']]" :key="p[0]" class="voice-chip">
            <button class="play" title="Hear this line at this speed" @click="tryIt('pace:' + p[0], { type: 'set_pace', lines: [sample.id], pace: p[0] })">{{ icon('pace:' + p[0]) }}</button>
            <button class="pick" @click="plan(p[1] + ' for ' + ids.length + ' line(s)', [{ type: 'set_pace', lines: ids, pace: p[0] }])">{{ p[1] }}</button>
          </span>
        </div>
        <div class="label">Feeling</div>
        <div class="choice-row">
          <span v-for="s in styles" :key="s" class="voice-chip">
            <button class="play" title="Hear this line with this feeling" @click="tryIt('style:' + s, { type: 'set_style', lines: [sample.id], style: s })">{{ icon('style:' + s) }}</button>
            <button class="pick" @click="setStyle(s)">{{ feeling(s) }}</button>
          </span>
        </div>
      </div>

      <div v-if="open === 'language'" class="choices">
        <div class="choice-row">
          <span v-for="l in languages" :key="l.code" class="voice-chip">
            <button class="play" title="Hear this line in this language" @click="tryIt('lang:' + l.code, { type: 'set_language', lines: [sample.id], lang: l.code })">{{ icon('lang:' + l.code) }}</button>
            <button class="pick" @click="setLang(l)"><b>{{ l.native }}</b> <span class="hint">{{ l.en }}</span></button>
          </span>
        </div>
      </div>

      <div v-if="open === 'words'" class="choices words">
        <div class="selbar-describe">
          <input class="input sm" v-model="text" placeholder="e.g. “make it funnier”, “line 2 says Muthuvel, not Muthu”" @keydown.enter="describe()" />
          <button class="btn sm primary" :disabled="!text.trim()" @click="describe()">Plan →</button>
        </div>
        <div class="choice-row">
          <button v-for="x in ['More casual', 'Shorter', 'Funnier', 'More formal', 'Simpler words']" :key="x" class="choice sm" @click="describe(x)">{{ x }}</button>
        </div>
      </div>
    </div>`,
};
