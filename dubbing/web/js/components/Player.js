// The video player and the speaker timeline under it.
//
// Two <video> elements sit on top of each other, the dub and the original.
//   Dub / Original   only one is visible and playing; switching copies the
//                    time across, so the same moment is compared instantly.
//   Compare          both are visible, split by a handle you can drag: the
//                    original on the left, the dub on the right, in sync.
//                    You hear the dub. Made to show the lip sync.
// With more than one dubbed language, the languages appear next to
// "Original" and switching keeps the current moment.
//
// The timeline is a moment picker: a filmstrip of the video with one lane per
// speaker under it. Click to jump; drag across it — or press [ and ] at the
// playhead — to mark a moment, then drag its handles to adjust. Edges snap to
// line boundaries. The marked moment can be played on a loop, and the page
// offers changes for the lines inside it. Hovering shows that frame.

import { fmtTime, speakerColor, speakerName } from "../util.js";
import Avatar from "./Avatar.js";

export default {
  components: { Avatar },
  props: {
    media: { type: Object, required: true },
    lines: { type: Array, default: () => [] },
    speakers: { type: Array, default: () => [] },
    activeId: { type: Number, default: null },
    selected: { type: Array, default: () => [] },
    langs: { type: Array, default: () => [] },
    lang: { type: String, default: "" },
    banner: { type: String, default: "" }, // e.g. "Previewing version 2"
    changed: { type: Array, default: () => [] }, // lines the last change redid
    comments: { type: Array, default: () => [] }, // open review comments: on the film
  },
  emits: ["time", "select", "range", "lang", "comment"],
  data: () => ({
    mode: "dub", // dub | orig | split
    now: 0,
    duration: 0,
    raf: 0,
    split: 50, // % of the width showing the original in Compare
    playing: false,
    stopAt: null, // end of a single line being played
    drag: null, // pointer down on the timeline: {x0, x1, rect, moved, line}
    sel: null, // the marked moment, seconds: {a, b}
    grab: null, // the range handle being dragged: "a" | "b"
    loop: false, // play the marked moment over and over
    hover: null, // {x (0-1), t} under the pointer, for the frame preview
    strip: false, // the filmstrip image loaded
    cc: (() => {
      try {
        return localStorage.getItem("cc") || "dub";
      } catch {
        return "dub";
      }
    })(), // subtitles on the video: off | dub | orig | both
    ccOpen: false, // the subtitle style popover is open
    ccStyle: (() => {
      try {
        return { size: 100, bg: 70, pos: 0, ...JSON.parse(localStorage.getItem("cc-style") || "{}") };
      } catch {
        return { size: 100, bg: 70, pos: 0 };
      }
    })(), // size %, background opacity %, raised by % of the video height
  }),
  computed: {
    total() {
      const last = this.lines.reduce((m, l) => Math.max(m, l.end), 0);
      return this.duration || last || 1;
    },
    tracks() {
      return this.speakers.map((s) => ({
        ...s,
        color: speakerColor(s.id),
        name: speakerName(s.id),
        lines: this.lines.filter((l) => l.speaker === s.id),
      }));
    },
    marks() {
      const step = [10, 15, 30, 60, 120, 300, 600].find((s) => this.total / s <= 8) || 900;
      const out = [];
      for (let t = 0; t <= this.total; t += step) out.push(t);
      return out;
    },
    current() {
      return this.lines.find((l) => this.now >= l.start && this.now < l.end) || null;
    },
    // What is drawn as the marked moment: the drag in progress, else sel.
    shown() {
      const d = this.drag;
      if (d && d.moved) {
        const [a, b] = [d.x0, d.x1].sort((x, y) => x - y);
        return { a: a * this.total, b: b * this.total };
      }
      return this.sel;
    },
    stripUrl() {
      return this.media.original.replace(/video\.mp4$/, "strip.jpg");
    },
    inside() {
      const s = this.sel;
      return s ? this.lines.filter((l) => l.end > s.a && l.start < s.b) : [];
    },
    hoverLine() {
      const h = this.hover;
      return h ? this.lines.find((l) => h.t >= l.start && h.t < l.end) : null;
    },
  },
  watch: {
    // The page cleared the selection: the marked moment goes too.
    selected(ids) {
      if (!ids.length && !this.grab) {
        this.sel = null;
        this.loop = false;
      }
    },
    stripUrl: {
      immediate: true,
      handler(url) {
        const img = new Image();
        img.onload = () => (this.strip = true);
        img.onerror = () => (this.strip = false);
        img.src = url;
      },
    },
    "media.dubbed"() {
      // A re-rendered dub, another language or a version: reload it at the
      // same moment, still playing if it was.
      const t = this.now;
      const wasPlaying = this.playing;
      this.$nextTick(() => {
        const v = this.$refs.dub;
        if (!v) return;
        v.addEventListener(
          "loadedmetadata",
          () => {
            v.currentTime = t;
            if (wasPlaying && this.mode !== "orig") v.play().catch(() => {});
          },
          { once: true },
        );
      });
    },
  },
  mounted() {
    window.addEventListener("keydown", this.onKey);
    const tick = () => {
      const v = this.video();
      if (v) {
        if (!v.paused) this.update(v);
        this.playing = !v.paused;
        if (this.stopAt != null && v.currentTime >= this.stopAt) {
          v.pause();
          this.stopAt = null;
        }
        if (this.loop && this.sel && !v.paused && v.currentTime >= this.sel.b) {
          v.currentTime = this.sel.a;
        }
        if (this.mode === "split") this.keepInSync();
      }
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  },
  unmounted() {
    window.removeEventListener("keydown", this.onKey);
    cancelAnimationFrame(this.raf);
  },
  methods: {
    fmtTime,
    // The video that is heard and drives the time (the dub in Compare).
    video() {
      return this.mode === "orig" ? this.$refs.orig : this.$refs.dub;
    },
    update(v) {
      this.now = v.currentTime;
      this.$emit("time", this.now);
    },
    onMeta(e) {
      this.duration = Math.max(this.duration, e.target.duration || 0);
    },
    switchTo(mode) {
      if (mode === this.mode) return;
      const dub = this.$refs.dub;
      const orig = this.$refs.orig;
      const from = this.video();
      const playing = from && !from.paused;
      dub.pause();
      orig.pause();
      const t = from.currentTime;
      dub.currentTime = orig.currentTime = t;
      // In Compare the original is shown but not heard; otherwise only one
      // video plays at a time. Volume carries over either way.
      orig.muted = mode === "split" ? true : dub.muted;
      orig.volume = dub.volume;
      this.mode = mode;
      if (playing) this.togglePlay(true);
    },
    keepInSync() {
      const dub = this.$refs.dub;
      const orig = this.$refs.orig;
      if (dub.paused !== orig.paused) dub.paused ? orig.pause() : orig.play().catch(() => {});
      if (Math.abs(orig.currentTime - dub.currentTime) > 0.12) orig.currentTime = dub.currentTime;
    },
    togglePlay(forcePlay) {
      const v = this.video();
      if (!v) return;
      if (forcePlay === true || v.paused) v.play().catch(() => {});
      else v.pause();
    },
    // Called by the page: jump to a time, optionally playing.
    seek(t, play = true) {
      const v = this.video();
      if (!v) return;
      v.currentTime = Math.max(0, t);
      if (this.mode === "split") this.$refs.orig.currentTime = v.currentTime;
      this.update(v);
      if (play) v.play().catch(() => {});
    },
    // Play just one line, in the original or the dub, then stop.
    playSpan(start, end, which) {
      if (which === "orig" && this.mode !== "orig") this.switchTo("orig");
      if (which === "dub" && this.mode === "orig") this.switchTo("dub");
      this.stopAt = end;
      this.seek(start, true);
    },
    pct(t) {
      return (100 * t) / this.total + "%";
    },

    // ── Timeline: click to jump, drag to mark a moment ───────────────────
    lanes() {
      return this.$refs.lanes.getBoundingClientRect();
    },
    at(e, rect) {
      return Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    },
    onDown(e) {
      if (e.button !== 0 || !this.$refs.lanes) return;
      const rect = this.lanes();
      if (e.clientX < rect.left) return; // the speaker names
      const x = this.at(e, rect);
      this.drag = { x0: x, x1: x, rect, moved: false, line: e.target.dataset.line };
      window.addEventListener("pointermove", this.onMove);
      window.addEventListener("pointerup", this.onUp, { once: true });
    },
    onMove(e) {
      const d = this.drag;
      if (!d) return;
      d.x1 = this.at(e, d.rect);
      if (Math.abs(d.x1 - d.x0) * d.rect.width > 5) d.moved = true;
    },
    onUp() {
      window.removeEventListener("pointermove", this.onMove);
      const d = this.drag;
      this.drag = null;
      if (!d) return;
      if (d.moved) {
        const [a, b] = [d.x0, d.x1].sort((x, y) => x - y);
        this.mark(a * this.total, b * this.total);
      } else {
        if (d.line != null) this.$emit("select", Number(d.line));
        this.seek(d.x0 * this.total, false);
      }
    },
    // Drag one edge of the marked moment.
    onGrab(edge, e) {
      const rect = this.lanes();
      this.grab = edge;
      const move = (ev) => {
        const t = this.at(ev, rect) * this.total;
        const s = { ...this.sel };
        s[edge] = t;
        if (s.b - s.a < 0.2) return;
        this.sel = s;
        this.seek(edge === "a" ? s.a : Math.max(s.a, s.b - 0.05), false); // show that frame
      };
      move(e);
      window.addEventListener("pointermove", move);
      window.addEventListener(
        "pointerup",
        () => {
          window.removeEventListener("pointermove", move);
          this.grab = null;
          this.mark(this.sel.a, this.sel.b);
        },
        { once: true },
      );
    },
    // Set the marked moment (edges snap to a line's start or end nearby) and
    // tell the page which stretch it covers.
    mark(a, b) {
      const snap = (t) => {
        let best = t;
        let gap = Math.max(0.35, this.total * 0.004);
        for (const l of this.lines)
          for (const x of [l.start, l.end])
            if (Math.abs(x - t) < gap) {
              gap = Math.abs(x - t);
              best = x;
            }
        return best;
      };
      a = Math.max(0, snap(a));
      b = Math.min(this.total, snap(b));
      if (b - a < 0.2) b = Math.min(this.total, a + 0.2);
      this.sel = { a, b };
      this.$emit("range", [a, b]);
    },
    markStart() {
      const t = this.now;
      const b = this.sel && this.sel.b > t + 0.2 ? this.sel.b : Math.min(this.total, t + 5);
      this.mark(t, b);
    },
    markEnd() {
      const t = this.now;
      const a = this.sel && this.sel.a < t - 0.2 ? this.sel.a : Math.max(0, t - 5);
      this.mark(a, t);
    },
    saveCcStyle() {
      try {
        localStorage.setItem("cc-style", JSON.stringify(this.ccStyle));
      } catch {
        /* not remembered */
      }
    },
    nextCc() {
      const order = ["dub", "orig", "both", "off"];
      this.cc = order[(order.indexOf(this.cc) + 1) % order.length];
      try {
        localStorage.setItem("cc", this.cc);
      } catch {
        /* not remembered */
      }
    },
    clearMark() {
      this.sel = null;
      this.loop = false;
      this.$emit("range", null);
    },
    playMoment() {
      if (!this.sel) return;
      if (this.loop && this.playing) {
        this.loop = false;
        this.togglePlay();
        return;
      }
      this.loop = true;
      this.seek(this.sel.a, true);
    },
    onHover(e) {
      if (!this.$refs.lanes || this.drag || this.grab) return (this.hover = null);
      const rect = this.lanes();
      if (e.clientX < rect.left || e.clientX > rect.right) return (this.hover = null);
      const x = this.at(e, rect);
      this.hover = { x, t: x * this.total };
    },
    // The filmstrip frame closest to a time (the strip holds 60 frames).
    frameStyle(t) {
      const i = Math.min(59, Math.max(0, Math.floor((t / this.total) * 60)));
      return {
        backgroundImage: `url(${this.stripUrl})`,
        backgroundSize: "6000% 100%",
        backgroundPosition: `${(i / 59) * 100}% 0`,
      };
    },

    // ── Compare handle ───────────────────────────────────────────────────
    onHandle(e) {
      const rect = this.$el.querySelector(".screen").getBoundingClientRect();
      const move = (ev) => (this.split = Math.min(95, Math.max(5, ((ev.clientX - rect.left) / rect.width) * 100)));
      move(e);
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", () => window.removeEventListener("pointermove", move), { once: true });
    },
    onKey(e) {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
      if (e.key === "[") this.markStart();
      if (e.key === "]") this.markEnd();
      if ((e.key === "l" || e.key === "L") && this.sel) this.playMoment();
      if (e.key === "Escape" && this.sel) this.clearMark();
      if (e.key === "o" || e.key === "O") this.switchTo(this.mode === "orig" ? "dub" : "orig");
      if (e.key === "c" || e.key === "C") this.switchTo(this.mode === "split" ? "dub" : "split");
      if (e.key === " ") {
        e.preventDefault();
        this.togglePlay();
      }
    },
  },
  template: `
    <div class="stage-col">
      <div class="screen" :class="'mode-' + mode">
        <div class="screen-top">
          <div class="segmented ab" title="Shortcut: O">
            <button :class="{ on: mode === 'orig' }" @click="switchTo('orig')">Original</button>
            <template v-if="langs.length > 1">
              <button v-for="l in langs" :key="l.code" :class="{ on: mode !== 'orig' && l.code === lang }"
                      @click="l.code === lang ? switchTo('dub') : $emit('lang', l.code)">{{ l.name }}</button>
            </template>
            <button v-else :class="{ on: mode === 'dub' }" @click="switchTo('dub')">Dub</button>
          </div>
          <button class="segmented ab compare-btn cc-btn" :class="{ on: cc !== 'off' }" :title="'Subtitles: ' + ({ dub: 'dub language', orig: 'original', both: 'both', off: 'off' })[cc] + ' — click to change'" @click="nextCc">
            <span>CC {{ ({ dub: lang.toUpperCase(), orig: 'Orig', both: 'Both', off: 'Off' })[cc] }}</span></button>
          <button class="segmented ab compare-btn" :class="{ on: ccOpen }" title="Subtitle size, background and position" @click="ccOpen = !ccOpen"><span>Aa</span></button>
          <div v-if="ccOpen" class="cc-pop" @pointerdown.stop>
            <label>Size <input type="range" min="70" max="180" step="5" v-model.number="ccStyle.size" @change="saveCcStyle" /><b>{{ ccStyle.size }}%</b></label>
            <label>Background <input type="range" min="0" max="100" step="5" v-model.number="ccStyle.bg" @change="saveCcStyle" /><b>{{ ccStyle.bg }}%</b></label>
            <label>Position <input type="range" min="0" max="60" step="2" v-model.number="ccStyle.pos" @change="saveCcStyle" /><b>{{ ccStyle.pos ? '↑' + ccStyle.pos + '%' : 'bottom' }}</b></label>
            <button class="btn ghost sm" @click="ccStyle = { size: 100, bg: 70, pos: 0 }; saveCcStyle()">Reset</button>
          </div>
          <button class="segmented ab compare-btn" :class="{ on: mode === 'split' }" title="Side by side — shortcut: C"
                  @click="switchTo(mode === 'split' ? 'dub' : 'split')"><span>⇆ Compare</span></button>
        </div>
        <div v-if="banner" class="screen-banner">{{ banner }}</div>

        <video ref="orig" v-show="mode !== 'dub'" :src="media.original" :controls="mode === 'orig'" playsinline preload="metadata"
               @timeupdate="mode === 'orig' && update($event.target)" @seeked="mode === 'orig' && update($event.target)" @loadedmetadata="onMeta"></video>
        <video ref="dub" v-show="mode !== 'orig'" :src="media.dubbed" :controls="mode === 'dub'" playsinline preload="auto"
               :style="mode === 'split' ? { clipPath: 'inset(0 0 0 ' + split + '%)' } : null"
               @timeupdate="mode !== 'orig' && update($event.target)" @seeked="mode !== 'orig' && update($event.target)" @loadedmetadata="onMeta"></video>

        <template v-if="mode === 'split'">
          <div class="split-handle" :style="{ left: split + '%' }" @pointerdown.prevent="onHandle"><span>⇆</span></div>
          <div class="split-label left">Original</div>
          <div class="split-label right">Dub</div>
          <div class="split-controls">
            <button class="btn sm" @click="togglePlay()">{{ playing ? '❚❚ Pause' : '▶ Play' }}</button>
            <span>{{ fmtTime(now) }} / {{ fmtTime(total) }}</span>
          </div>
        </template>
        <div v-if="mode !== 'split' && current && cc !== 'off'" class="now-line"
             :style="{ fontSize: ccStyle.size + '%', '--cc-bg': ccStyle.bg / 100, transform: 'translateY(-' + ccStyle.pos + 'cqh)' }">
          <span v-if="(cc === 'orig' || cc === 'both') && current.text" class="orig">{{ current.text }}</span>
          <span v-if="(cc === 'dub' || cc === 'both') && current.dubbed && !current.kept">{{ current.dubbed }}</span>
        </div>
      </div>

      <div class="panel timeline">
        <div class="timeline-head">
          <span class="tl-title"><b>Moment picker</b> <span class="hint">— drag across the film to pick the part you want to change</span></span>
          <span class="keys"><kbd>Space</kbd> play · <kbd>[</kbd> <kbd>]</kbd> mark · <kbd>L</kbd> loop · <kbd>O</kbd> original · <kbd>C</kbd> compare</span>
        </div>
        <div class="tl" @pointerdown="onDown" @pointermove="onHover" @pointerleave="hover = null">
          <div class="tl-names">
            <div class="tl-name film-name"><span class="full">Video</span></div>
            <div v-for="t in tracks" :key="t.id" class="tl-name"><Avatar :speaker="t" :size="22" /><span class="full">{{ t.name }}</span></div>
          </div>
          <div class="tl-lanes" ref="lanes">
            <div class="film" :class="{ loaded: strip }" :style="strip ? { backgroundImage: 'url(' + stripUrl + ')' } : null"></div>
            <span v-for="c in comments" :key="c.id" class="cmark" :style="{ left: pct(c.t) }" :title="c.name + ': ' + c.text"
                  @pointerdown.stop @click.stop="$emit('comment', c)"></span>
            <div v-for="t in tracks" :key="t.id" class="lane">
              <div v-for="l in t.lines" :key="l.id" :data-line="l.id" class="block"
                   :class="{ active: l.id === activeId, kept: l.kept, selected: selected.includes(l.id), changed: changed.includes(l.id) }"
                   :style="{ left: pct(l.start), width: pct(l.end - l.start), background: t.color }"></div>
            </div>
            <div v-if="shown" class="range" :class="{ live: drag }" :style="{ left: pct(shown.a), width: pct(shown.b - shown.a) }">
              <div v-if="!drag" class="handle a" @pointerdown.stop.prevent="onGrab('a', $event)" title="Drag to move the start"><span>{{ fmtTime(shown.a) }}</span></div>
              <div v-if="!drag" class="handle b" @pointerdown.stop.prevent="onGrab('b', $event)" title="Drag to move the end"><span>{{ fmtTime(shown.b) }}</span></div>
            </div>
            <div class="playhead" :style="{ left: pct(now) }"><i></i></div>
            <div v-if="hover" class="scrub" :style="{ left: hover.x * 100 + '%' }">
              <div class="scrub-card" :class="{ right: hover.x > 0.82, left: hover.x < 0.18 }">
                <div v-if="strip" class="scrub-frame" :style="frameStyle(hover.t)"></div>
                <div class="scrub-time">{{ fmtTime(hover.t) }}</div>
                <div v-if="hoverLine" class="scrub-line">{{ hoverLine.dubbed || hoverLine.text }}</div>
              </div>
            </div>
          </div>
        </div>
        <div class="ruler">
          <span></span>
          <div class="ruler-marks"><span v-for="m in marks" :key="m" :style="{ left: pct(m) }">{{ fmtTime(m) }}</span></div>
        </div>
        <div class="moment-bar" :class="{ has: sel }">
          <template v-if="sel">
            <button class="btn sm primary" @click="playMoment">{{ loop && playing ? '❚❚ Stop' : 'Play this moment' }}</button>
            <span class="moment-time"><b>{{ fmtTime(sel.a) }}</b> → <b>{{ fmtTime(sel.b) }}</b> <span class="hint">· {{ (sel.b - sel.a).toFixed(1) }}s · {{ inside.length }} line{{ inside.length === 1 ? '' : 's' }}</span></span>
            <span class="spacer"></span>
            <button class="btn ghost sm" title="Start the moment at the playhead — shortcut [" @click="markStart">⇤ Start here</button>
            <button class="btn ghost sm" title="End the moment at the playhead — shortcut ]" @click="markEnd">End here ⇥</button>
            <button class="btn ghost icon sm" title="Clear — Esc" @click="clearMark">✕</button>
          </template>
          <template v-else>
            <span class="hint">Tip: pause where something should change, then</span>
            <button class="btn ghost sm" @click="markStart">⇤ Mark start here</button>
            <button class="btn ghost sm" @click="markEnd">Mark end here ⇥</button>
          </template>
        </div>
      </div>
    </div>`,
};
