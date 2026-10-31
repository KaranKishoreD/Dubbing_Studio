// "Use my voice" for one speaker: record about 20 seconds in the browser
// (or pick a recording), hear it back, say whether it is a man's or a
// woman's voice, and add it to the waiting changes. That speaker's lines are
// then cloned from this recording (pipeline/voice/voice_clone.py).

import { api } from "../api.js";
import { speakerName } from "../util.js";

const READ_THIS =
  "Hi! I'm recording a short sample of my voice. I'll speak naturally, the way I usually talk — " +
  "not too fast and not too slow. One, two, three, four, five. The weather is lovely today, " +
  "and I'm looking forward to hearing this video in a brand new language.";

export default {
  props: {
    projectId: { type: String, required: true },
    speaker: { type: Object, required: true },
  },
  emits: ["close", "plan"],
  data() {
    return {
      READ_THIS,
      state: "idle", // idle | recording | review | saving
      secs: 0,
      timer: 0,
      rec: null,
      blob: null,
      url: "",
      gender: this.speaker.gender === "female" ? "female" : "male",
      error: "",
    };
  },
  unmounted() {
    clearInterval(this.timer);
    this.rec?.stream?.getTracks().forEach((t) => t.stop());
  },
  methods: {
    speakerName,
    async start() {
      this.error = "";
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
        const chunks = [];
        this.rec = new MediaRecorder(stream);
        this.rec.ondataavailable = (e) => chunks.push(e.data);
        this.rec.onstop = () => {
          stream.getTracks().forEach((t) => t.stop());
          this.use(new Blob(chunks, { type: this.rec.mimeType }));
        };
        this.rec.start();
        this.secs = 0;
        this.state = "recording";
        this.timer = setInterval(() => {
          this.secs += 1;
          if (this.secs >= 30) this.stop();
        }, 1000);
      } catch {
        this.error = "The browser could not use your microphone. Allow it, or upload a recording instead.";
      }
    },
    stop() {
      clearInterval(this.timer);
      if (this.rec && this.rec.state !== "inactive") this.rec.stop();
    },
    use(blob) {
      this.blob = blob;
      this.url = URL.createObjectURL(blob);
      this.state = "review";
    },
    pick(e) {
      const f = e.target.files[0];
      if (f) {
        this.secs = 0;
        this.use(f);
      }
    },
    async save() {
      this.state = "saving";
      try {
        const r = await api.myVoice(this.projectId, this.speaker.id, this.blob);
        const label = `${speakerName(this.speaker.id)} speaks in your voice`;
        this.$emit("plan", {
          requests: [{ request: label, actions: [{ type: "own_voice", speaker: this.speaker.id + 1, ref: r.ref, gender: this.gender }] }],
        });
        this.$emit("close");
      } catch (e) {
        this.error = e.message;
        this.state = "review";
      }
    },
  },
  template: `
    <div class="backdrop" @click.self="$emit('close')">
      <div class="dialog" style="width:560px">
        <div class="dialog-head">
          <h2>Your voice for {{ speakerName(speaker.id) }}</h2>
          <span style="flex:1"></span>
          <button class="btn ghost icon" aria-label="Close" @click="$emit('close')">✕</button>
        </div>
        <div class="dialog-body">
          <p class="hint" style="margin-top:0">Read this out loud in your normal voice, somewhere quiet. 15–30 seconds is perfect. Your lines are then spoken in the dub's language — in your voice.</p>
          <div class="read-this">{{ READ_THIS }}</div>

          <div class="rec-row">
            <button v-if="state !== 'recording'" class="btn primary lg" @click="start">● {{ state === 'review' ? 'Record again' : 'Start recording' }}</button>
            <button v-else class="btn danger lg" @click="stop">■ Stop · {{ secs }}s</button>
            <span v-if="state === 'recording'" class="rec-dot"></span>
            <span class="hint">or <a href="#" @click.prevent="$refs.file.click()">upload a recording</a></span>
            <input ref="file" type="file" accept="audio/*" hidden @change="pick" />
          </div>

          <template v-if="state === 'review' || state === 'saving'">
            <audio :src="url" controls style="width:100%;margin-top:12px"></audio>
            <div class="field" style="margin-top:12px">
              <label>This voice is a…</label>
              <div class="choice-row">
                <button class="choice" :class="{ on: gender === 'male' }" @click="gender = 'male'">Man's voice</button>
                <button class="choice" :class="{ on: gender === 'female' }" @click="gender = 'female'">Woman's voice</button>
              </div>
              <div v-if="speaker.gender && gender !== speaker.gender" class="hint" style="margin-top:6px">{{ speakerName(speaker.id) }} was a {{ speaker.gender === 'male' ? 'man' : 'woman' }} — their lines will be reworded to match (in Hindi or Tamil, verbs change with gender).</div>
            </div>
          </template>
          <div v-if="error" class="note error" style="margin-top:10px">{{ error }}</div>
        </div>
        <div class="dialog-foot" style="display:flex;justify-content:flex-end;gap:8px;padding:0 22px 20px">
          <button class="btn" @click="$emit('close')">Cancel</button>
          <button class="btn primary" :disabled="state !== 'review'" @click="save">
            <span v-if="state === 'saving'" class="spinner"></span> Add to changes
          </button>
        </div>
      </div>
    </div>`,
};
