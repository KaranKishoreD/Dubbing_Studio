// Downloads: the dubbed video, the soundtrack alone, and subtitles.

import { api } from "../api.js";
import { fmtSize } from "../util.js";

const KINDS = {
  mp4: ["film", "Dubbed video", "MP4 · H.264 + AAC, plays everywhere"],
  m4a: ["headphones", "Dubbed soundtrack", "M4A · audio only"],
  srt: ["captions", "Subtitles", "SRT"],
};

export default {
  props: { projectId: { type: String, required: true }, lang: { type: String, required: true }, editable: Boolean, stamp: [String, Number] },
  data: () => ({ files: [], loading: true, error: "", burning: false, burned: null }),
  watch: {
    lang: "load",
    stamp: "load",
  },
  mounted() {
    this.load();
  },
  methods: {
    fmtSize,
    async load() {
      this.loading = true;
      this.error = "";
      try {
        this.files = this.editable ? await api.exports(this.projectId, this.lang) : [];
      } catch (e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },
    // The dubbed video with subtitles burned in, made on request.
    async burn() {
      this.burning = true;
      try {
        this.burned = await api.subtitled(this.projectId, this.lang);
        const a = document.createElement("a");
        a.href = this.burned.url;
        a.download = this.burned.url.split("/").pop();
        a.click();
      } catch (e) {
        this.error = e.message;
      } finally {
        this.burning = false;
      }
    },
    describe(f) {
      const ext = f.name.split(".").pop();
      const [icon, title, sub] = KINDS[ext] || ["file", f.name, ""];
      const which = ext === "srt" ? (f.name.includes("original") ? " — original language" : " — dub language") : "";
      return { icon, title: title + which, sub };
    },
  },
  template: `
    <div class="tab-pad">
      <div v-if="loading" class="hint">Preparing files…</div>
      <div v-else-if="error" class="note error">{{ error }}</div>
      <div v-else-if="!files.length" class="hint">Nothing to download yet.</div>
      <a v-for="f in files" :key="f.name" class="file" :href="f.url" :download="f.name">
        <div class="icon"><Icon :name="describe(f).icon" /></div>
        <div class="meta"><div>{{ describe(f).title }}</div><div>{{ describe(f).sub }} · {{ fmtSize(f.size) }}</div></div>
        <span class="btn sm">Download</span>
      </a>
      <div v-if="files.length" class="file" style="cursor:pointer" @click="!burning && burn()">
        <div class="icon"><Icon name="captions" /></div>
        <div class="meta"><div>Dubbed video with subtitles burned in</div><div>MP4 · subtitles always shown — good for social media{{ burned ? ' · ' + fmtSize(burned.size) : '' }}</div></div>
        <span class="btn sm"><span v-if="burning" class="spinner"></span>{{ burning ? ' Making…' : 'Download' }}</span>
      </div>
    </div>`,
};
