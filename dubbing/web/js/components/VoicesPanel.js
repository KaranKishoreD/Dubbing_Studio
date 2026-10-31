// Which voice each speaker gets. Pick, preview, then re-voice: only the
// changed speakers' lines are spoken again; translations are kept.

import { store } from "../store.js";
import { speakerName } from "../util.js";
import Avatar from "./Avatar.js";
import VoiceRecorder from "./VoiceRecorder.js";

export default {
  components: { Avatar, VoiceRecorder },
  props: {
    speakers: { type: Array, required: true },
    lang: { type: String, required: true },
    projectId: { type: String, required: true },
    editable: Boolean,
    busy: Boolean,
  },
  emits: ["revoice", "plan"],
  data: () => ({ picks: {}, playing: null, audio: null, people: "", recording: null }),
  computed: {
    voices: () => store.options?.voices || [],
    changed() {
      return Object.entries(this.picks).filter(([id, v]) => v && v !== this.current(Number(id)));
    },
  },
  watch: {
    speakers: {
      immediate: true,
      handler() {
        this.picks = Object.fromEntries(this.speakers.map((s) => [s.id, s.voice]));
      },
    },
  },
  unmounted() {
    this.audio?.pause();
  },
  methods: {
    speakerName,
    // Fixes that change who the speakers are, applied like any other edit.
    fix(label, action) {
      this.$emit("plan", { requests: [{ request: label, actions: [action] }] }, label);
    },
    flipGender(s) {
      const gender = s.gender === "male" ? "female" : "male";
      this.fix(`${speakerName(s.id)} is ${gender}`, { type: "set_gender", speaker: s.id + 1, gender });
    },
    merge(s, e) {
      const into = e.target.value;
      e.target.value = "";
      if (into === "") return;
      this.fix(`${speakerName(s.id)} is the same person as ${speakerName(Number(into))}`, {
        type: "merge_speakers",
        speaker: s.id + 1,
        into: Number(into) + 1,
      });
    },
    steady(s) {
      this.fix(`One steady voice for ${speakerName(s.id)}`, { type: "steady_voice", speaker: s.id + 1 });
    },
    redetect() {
      const n = Number(this.people) || null;
      const ask =
        "Find who speaks each line again" + (n ? ` with ${n} people` : "") +
        " and dub the whole video again? Changes made since the first dub are replaced; the current version stays in History.";
      if (confirm(ask)) this.fix("Find speakers again" + (n ? ` (${n} people)` : ""), { type: "redetect_speakers", people: n });
    },
    current(id) {
      return this.speakers.find((s) => s.id === id)?.voice;
    },
    byGender(g) {
      return this.voices.filter((v) => v.gender === g);
    },
    preview(id) {
      const voice = this.picks[id];
      if (!voice) return;
      this.audio?.pause();
      this.playing = id;
      this.audio = new Audio(`/api/voices/preview?voice=${encodeURIComponent(voice)}&lang=${this.lang}`);
      this.audio.onended = this.audio.onerror = () => (this.playing = null);
      this.audio.play().catch(() => (this.playing = null));
    },
    apply() {
      this.$emit("revoice", Object.fromEntries(this.changed.map(([id, v]) => [String(id), v])));
    },
  },
  template: `
    <div class="tab-pad">
      <p class="hint" style="margin:0 0 14px">Each speaker keeps one voice for the whole video. Choose another and listen. Every change here waits in your list of changes until you apply them all together. To split a speaker into two people, select their lines on the timeline.</p>
      <div v-for="s in speakers" :key="s.id" class="speaker-card" :class="{ changed: picks[s.id] && picks[s.id] !== s.voice }">
        <div class="top">
          <Avatar :speaker="s" :size="36" />
          <div>
            <div class="name">{{ speakerName(s.id) }}</div>
            <div class="stats">{{ s.gender || 'unknown' }} · {{ s.lines }} lines · {{ Math.round(s.seconds) }}s</div>
          </div>
        </div>
        <div class="pick">
          <select class="input" v-model="picks[s.id]" :disabled="!editable || busy">
            <option v-if="!s.voice" :value="''">Automatic</option>
            <optgroup label="Female"><option v-for="v in byGender('female')" :key="v.value" :value="v.value">{{ v.label }}</option></optgroup>
            <optgroup label="Male"><option v-for="v in byGender('male')" :key="v.value" :value="v.value">{{ v.label }}</option></optgroup>
          </select>
          <button class="btn icon" :disabled="!picks[s.id]" :title="'Hear ' + (picks[s.id] || 'this voice')" @click="preview(s.id)">
            <span v-if="playing === s.id" class="spinner"></span><span v-else>▶</span>
          </button>
        </div>
        <div v-if="editable" class="fixes">
          <button class="btn ghost sm" title="Record your own voice for this speaker" @click="recording = s">Use my voice</button>
          <button class="btn ghost sm" :disabled="busy" title="Their voice changes from line to line or sounds like someone else: re-clone every line from their own clean voice" @click="steady(s)">Steady voice</button>
          <button class="btn ghost sm" :disabled="busy" title="The speaker's gender was detected wrong" @click="flipGender(s)">
            Actually {{ s.gender === 'male' ? 'a woman' : 'a man' }}
          </button>
          <select v-if="speakers.length > 1" class="input sm" :disabled="busy" title="Two speakers are the same person" @change="merge(s, $event)">
            <option value="">Same person as…</option>
            <option v-for="o in speakers.filter(o => o.id !== s.id)" :key="o.id" :value="o.id">{{ speakerName(o.id) }}</option>
          </select>
        </div>
      </div>
      <button class="btn primary" style="width:100%;margin-top:6px" :disabled="!editable || busy || !changed.length" @click="apply">
        Add {{ changed.length ? changed.length + ' voice change' + (changed.length > 1 ? 's' : '') : 'voice changes' }} to changes
      </button>
      <div v-if="editable" class="redetect">
        <span class="hint">Speakers mixed up throughout?</span>
        <select class="input sm" v-model="people" :disabled="busy" title="How many different people speak">
          <option value="">People: automatic</option>
          <option v-for="n in 8" :key="n" :value="n">{{ n }} {{ n > 1 ? 'people' : 'person' }}</option>
        </select>
        <button class="btn sm" :disabled="busy" @click="redetect">Find speakers again</button>
      </div>
      <VoiceRecorder v-if="recording" :project-id="projectId" :speaker="recording" @close="recording = null" @plan="$emit('plan', $event)" />
    </div>`,
};
