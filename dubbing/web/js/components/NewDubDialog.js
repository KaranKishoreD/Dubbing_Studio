// The "New dub" dialog: a video link (any site yt-dlp supports) or a file
// from the computer, a language, optionally a part of the video.
// Everything else has sensible defaults under "Advanced".

import { api } from "../api.js";
import { store, track, go, toast } from "../store.js";
import { splitLangLabel, youtubeId, fmtLeft, finishAt } from "../util.js";

export default {
  props: { presets: { type: Object, default: () => ({}) } },
  emits: ["close"],
  data() {
    // Presets: a link and range ("+ Language"), or every field of a stopped
    // dub's form ("Change settings & start again").
    const p = this.presets;
    return {
      form: {
        url: p.url || "",
        lang: p.lang || "hi",
        // "indic": Bodhan/Sarvam, the 22 Indian languages; "international":
        // LLM translation and Kokoro's open voices for world languages.
        mode: p.mode || "indic",
        start: p.start || "",
        end: p.end || "",
        speakers: p.speakers ?? null,
        lang_mix: p.lang_mix || "standard",
        translator: p.translator || "indic",
        songs: p.songs || "keep",
        engine: p.engine || "bodhan",
        domain: p.domain || "education / lecture",
        style: p.style || store.options?.default_style || "",
        lipsync: p.lipsync ?? true,
      },
      sending: false,
      error: "",
      several: false, // dub into more than one language at once
      extra: [], // the other languages ticked in "several" mode
      probe: null, // what the link points to: {ok, title, thumbnail, site} | {ok: false, error}
      probing: false,
      upload: null, // {name, progress, info} while / after a file uploads
      dragOver: false,
      estimate: null, // {seconds, notes} for the chosen clip, from past runs
    };
  },
  computed: {
    // The whole video's length in seconds, once the link check or the
    // upload has told us (null until then).
    knownLength() {
      const d = this.upload?.info?.duration ?? (this.probe?.ok ? this.probe.duration : null);
      return d > 0 ? d : null;
    },
    options: () => store.options,
    videoId() {
      return youtubeId(this.form.url);
    },
    isLink() {
      return /^https?:\/\/\S+\.\S+/.test(this.form.url);
    },
    ready() {
      if (this.upload) return !!this.upload.info;
      return this.isLink && !(this.probe && this.probe.ok === false);
    },
    languages() {
      const list = this.form.mode === "international" ? this.options?.international : this.options?.languages;
      return (list || []).map((l) => ({ code: l.code, ...splitLangLabel(l.label) }));
    },
  },
  watch: {
    "form.url"(url) {
      clearTimeout(this.probeTimer);
      this.probe = null;
      if (!this.isLink || this.videoId) return; // YouTube shows its picture at once
      this.probing = true;
      this.probeTimer = setTimeout(async () => {
        try {
          if (this.form.url === url) this.probe = await api.probe(url);
        } catch {
          this.probe = null;
        } finally {
          this.probing = false;
        }
      }, 500);
    },
    "form.start": "estimateTime",
    "form.end": "estimateTime",
    "form.lipsync": "estimateTime",
    knownLength: "estimateTime",
  },
  mounted() {
    this.estimateTime();
    this.$refs.url?.focus();
    if (this.presets.file) this.sendFile(this.presets.file); // dropped on the home page
    window.addEventListener("keydown", this.onKey);
  },
  unmounted() {
    window.removeEventListener("keydown", this.onKey);
  },
  methods: {
    // Switching mode picks that mode's first language (and keeps several-
    // language picks within the mode).
    setMode(m) {
      if (this.form.mode === m) return;
      this.form.mode = m;
      this.extra = [];
      this.form.lang = m === "international" ? "en" : "hi";
    },
    onKey(e) {
      if (e.key === "Escape") this.$emit("close");
    },
    // With a clip range, or the length from the link check or the upload, the
    // dialog can say how long the dub should take on this machine right now.
    async estimateTime() {
      const secs = (s) => (s ? s.split(":").reduce((a, x) => a * 60 + Number(x), 0) : null);
      const a = secs(this.form.start) ?? 0;
      const b = secs(this.form.end) ?? this.knownLength;
      if (b == null || isNaN(a) || isNaN(b) || b <= a) return (this.estimate = null);
      try {
        this.estimate = await api.estimate(b - a, this.form.lipsync);
      } catch {
        this.estimate = null;
      }
    },
    fmtLeft,
    finishAt,
    pickLang(code) {
      if (!this.several) return (this.form.lang = code);
      if (code === this.form.lang) {
        // the first one can only be unticked if another stays
        if (this.extra.length) [this.form.lang, ...this.extra] = this.extra;
        return;
      }
      this.extra = this.extra.includes(code) ? this.extra.filter((c) => c !== code) : [...this.extra, code];
    },
    pickFile() {
      this.$refs.file.click();
    },
    onDrop(e) {
      this.dragOver = false;
      const f = e.dataTransfer?.files?.[0];
      if (f) this.sendFile(f);
    },
    async sendFile(file) {
      if (!file) return;
      this.error = "";
      this.upload = { name: file.name, progress: 0, info: null };
      try {
        const info = await api.upload(file, (p) => (this.upload.progress = p));
        this.upload.info = info;
        this.form.url = info.source;
        if (info.audio_only) this.form.lipsync = false;
      } catch (e) {
        this.error = e.message;
        this.upload = null;
      }
    },
    clearUpload() {
      this.upload = null;
      this.form.url = "";
    },
    async start() {
      this.error = "";
      if (!this.ready) {
        this.error = this.probe?.error || "Paste a video link (YouTube, Vimeo, Instagram, X…) or upload a file.";
        return;
      }
      this.sending = true;
      try {
        // One dub per language. They run one after another; the steps that do
        // not depend on the language (download, voices apart, transcript,
        // speakers) are done once and reused.
        const langs = [this.form.lang, ...(this.several ? this.extra.filter((c) => c !== this.form.lang) : [])];
        const jobs = [];
        const duration = this.knownLength || null;
        for (const lang of langs) jobs.push(await api.dub({ ...this.form, lang, duration }));
        jobs.forEach(track);
        toast(langs.length > 1 ? `${langs.length} dubs queued — the first one is starting.` : "Dub started — you can keep browsing while it runs.");
        this.$emit("close");
        go(`#/job/${jobs[0].id}`);
      } catch (e) {
        this.error = e.message;
      } finally {
        this.sending = false;
      }
    },
  },
  template: `
    <div class="backdrop" @click.self="$emit('close')">
      <div class="dialog" role="dialog" aria-label="New dub">
        <div class="dialog-head">
          <h2>New dub</h2>
          <span style="flex:1"></span>
          <button class="btn ghost icon" @click="$emit('close')" aria-label="Close">✕</button>
        </div>
        <div class="dialog-body">
          <div class="mode-switch" role="radiogroup" aria-label="Dubbing mode">
            <button type="button" role="radio" :aria-checked="form.mode === 'indic'" :class="{ on: form.mode === 'indic' }" @click="setMode('indic')">
              <b>Indic</b><small>22 Indian languages · Indian voice models</small>
            </button>
            <button type="button" role="radio" :aria-checked="form.mode === 'international'" :class="{ on: form.mode === 'international' }" @click="setMode('international')">
              <b>International</b><small>English, Spanish, French, Italian, Portuguese, Japanese, Chinese · open models</small>
            </button>
          </div>
          <div class="field" @dragover.prevent="dragOver = true" @dragleave="dragOver = false" @drop.prevent="onDrop">
            <label for="url">Video</label>
            <template v-if="!upload">
              <input id="url" ref="url" class="input lg" v-model.trim="form.url" placeholder="Paste a link — YouTube, Vimeo, Instagram, X, Facebook…" @keydown.enter="start" />
              <div v-if="videoId" class="url-preview">
                <img :src="'https://i.ytimg.com/vi/' + videoId + '/mqdefault.jpg'" alt="" />
                <div class="hint">Video found. Pick a language, then start.</div>
              </div>
              <div v-else-if="probing" class="url-preview"><span class="spinner"></span><div class="hint">Checking the link…</div></div>
              <div v-else-if="probe && probe.ok" class="url-preview">
                <img v-if="probe.thumbnail" :src="probe.thumbnail" alt="" />
                <div><b>{{ probe.title }}</b><div class="hint">Found on {{ probe.site }}. Pick a language, then start.</div></div>
              </div>
              <div v-else-if="probe" class="note error" style="margin-top:8px">{{ probe.error }}</div>
              <div class="dropzone" :class="{ over: dragOver }" @click="pickFile">
                <b>Or drop a video or audio file here</b> <span class="hint">— or click to choose one (MP4, MOV, MKV, MP3, WAV…)</span>
              </div>
            </template>
            <div v-else class="url-preview">
              <span style="font-size:28px">{{ upload.info?.audio_only ? '' : '' }}</span>
              <div style="flex:1;min-width:0">
                <b class="ellipsis" style="display:block">{{ upload.name }}</b>
                <div v-if="!upload.info" class="upbar"><div :style="{ width: Math.round(upload.progress * 100) + '%' }"></div></div>
                <div v-else class="hint">Uploaded{{ upload.info.audio_only ? ' — audio only, so no lip sync needed' : '' }}. Pick a language, then start.</div>
              </div>
              <button class="btn ghost sm" @click="clearUpload">Change</button>
            </div>
            <input ref="file" type="file" accept="video/*,audio/*,.mkv,.mov" hidden @change="sendFile($event.target.files[0])" />
          </div>

          <div class="field">
            <label style="display:flex;align-items:center">Dub into
              <span style="flex:1"></span>
              <label class="switch-row sm" style="text-transform:none;font-weight:400;margin:0">
                <input type="checkbox" v-model="several" /> Several languages
              </label>
            </label>
            <div class="langs">
              <button v-for="l in languages" :key="l.code" type="button" class="lang-btn"
                      :class="{ on: form.lang === l.code || (several && extra.includes(l.code)) }" @click="pickLang(l.code)">
                <span class="native">{{ l.native }}</span><span class="en">{{ l.en }}</span>
              </button>
            </div>
            <div v-if="several" class="hint" style="margin-top:6px">Tick every language you want. They're made one after another, and the shared steps (download, transcript, speakers) are done only once.</div>
          </div>

          <details class="more">
            <summary>Advanced</summary>
            <div class="more-body">
              <div class="row2">
                <div class="field">
                  <label>Speakers</label>
                  <select class="input" v-model="form.speakers">
                    <option :value="null">Detect automatically</option>
                    <option v-for="n in 12" :key="n" :value="n">{{ n }}</option>
                  </select>
                </div>
                <div class="field">
                  <label>Content</label>
                  <select class="input" v-model="form.domain">
                    <option v-for="d in options.domains" :key="d" :value="d">{{ d }}</option>
                  </select>
                </div>
              </div>
              <div v-if="form.mode === 'indic'" class="field">
                <label>English words in the dub</label>
                <div class="segmented">
                  <button v-for="o in [['pure','None'],['standard','Standard'],['natural','Natural'],['heavy','Heavy mix']]" :key="o[0]" type="button" :class="{ on: form.lang_mix === o[0] }" @click="form.lang_mix = o[0]">{{ o[1] }}</button>
                </div>
                <div class="hint">Natural keeps the English words people really use ("phone", "meeting"); Heavy mix is Hinglish/Tanglish.</div>
              </div>
              <div v-if="form.mode === 'indic'" class="field">
                <label>Translation</label>
                <div class="segmented">
                  <button type="button" :class="{ on: form.translator === 'concise' }" @click="form.translator = 'concise'">Concise (short lines, no polish)</button>
                  <button type="button" :class="{ on: form.translator === 'indic' }" @click="form.translator = 'indic'">Indic + LLM polish</button>
                  <button type="button" :class="{ on: form.translator === 'llm' }" @click="form.translator = 'llm'">LLM only</button>
                </div>
              </div>
              <div class="field">
                <label>Songs</label>
                <div class="segmented">
                  <button type="button" :class="{ on: form.songs === 'keep' }" @click="form.songs = 'keep'">Keep original singing</button>
                  <button type="button" :class="{ on: form.songs === 'resing' }" @click="form.songs = 'resing'">Re-sing (experimental)</button>
                </div>
              </div>
              <div v-if="form.mode === 'indic'" class="row2">
                <div class="field">
                  <label>Voice engine</label>
                  <select class="input" v-model="form.engine">
                    <option value="bodhan">Bodhan Indic-Speak</option>
                    <option value="sarvam" :disabled="!options.sarvam">Sarvam Bulbul{{ options.sarvam ? '' : ' (add key)' }}</option>
                  </select>
                </div>
                <div class="field">
                  <label>Speaking style</label>
                  <select class="input" v-model="form.style">
                    <option v-for="s in options.styles" :key="s" :value="s">{{ s }}</option>
                  </select>
                </div>
              </div>
            </div>
          </details>

          <div class="field">
            <label>Only part of the video <span style="text-transform:none;font-weight:400">(optional)</span></label>
            <div class="row2">
              <input class="input" v-model.trim="form.start" placeholder="From, e.g. 1:30" />
              <input class="input" v-model.trim="form.end" placeholder="To, e.g. 3:00" />
            </div>
            <div class="hint">Only this part is downloaded and dubbed — a quick way to try a long video.</div>
          </div>

          <label v-if="!upload?.info?.audio_only" class="switch-row">
            <input type="checkbox" v-model="form.lipsync" />
            <span><b>Lip sync</b> — mouths move with the new language.
              <span class="hint">Off = fast mode: much quicker, mouths keep their original movement. You can add lip sync later.</span></span>
          </label>

          <div v-if="estimate" class="estimate">
            <div>This dub should take <b>{{ fmtLeft(estimate.seconds) }}</b> — ready around {{ finishAt(estimate.seconds) }}.</div>
            <div v-for="n in estimate.notes" :key="n" class="warn">{{ n }}</div>
          </div>



          <div v-if="error" class="note error">{{ error }}</div>
        </div>
        <div class="dialog-foot">
          <button class="btn" @click="$emit('close')">Cancel</button>
          <button class="btn primary lg" :disabled="sending || !ready" @click="start">
            <span v-if="sending" class="spinner"></span> Start dubbing
          </button>
        </div>
      </div>
    </div>`,
};
