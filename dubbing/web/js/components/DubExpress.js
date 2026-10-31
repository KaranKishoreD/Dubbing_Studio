// "How a dub is made": the pipeline's stages as the stops of one railway line.
// The line is a single unbroken track that snakes through the rows of station
// boards: it starts at the top right, runs left, curves round, runs right,
// and so on, ending at the last stop.
//
// The visitor drives: the train waits at the start, and a click anywhere on
// the panel (or "Next stop", or →) takes it one stop on. A board comes into focus when the train
// arrives; the stops still ahead are blurred, so what happens there stays a
// surprise. Nothing moves on its own and nothing loops.

const HOP_MS = 375; // one stop to the next
const START = 14; // where the train waits, along the track (px)
const CURVE = 26; // how far a turn bulges past the boards (px)

export default {
  props: { stages: { type: Array, required: true } }, // [{key, hi, en, place, code, words, engine}]
  data: () => ({
    cols: 5,
    w: 0,
    h: 0,
    d: "", // the track's SVG path
    stops: [], // path length at each stop
    shown: 0, // boards lit so far
    stop: -1, // the stop the train is at (-1: waiting at the start)
    cur: START, // the train's position along the track (px)
    moving: false,
    train: null, // {x, y, angle}
    raf: 0,
    still: matchMedia("(prefers-reduced-motion: reduce)").matches,
  }),
  computed: {
    trainStyle() {
      if (!this.train) return { display: "none" };
      const { x, y, angle } = this.train;
      // The engine is drawn facing right; mirror it when heading left, then
      // turn it along the track. Its wheels sit on the rails.
      const heading = Math.cos((angle * Math.PI) / 180) >= 0 ? `rotate(${angle}deg)` : `rotate(${angle - 180}deg) scaleX(-1)`;
      return { left: x + "px", top: y + "px", transform: `translate(-50%, -88%) ${heading}`, transformOrigin: "50% 88%" };
    },
  },
  mounted() {
    this.resize = new ResizeObserver(() => this.layout(false));
    this.resize.observe(this.$el);
    this.layout();
  },
  unmounted() {
    this.resize?.disconnect();
    cancelAnimationFrame(this.raf);
  },
  methods: {
    // Boards go right-to-left on even rows and left-to-right on odd rows, so
    // the order follows the track.
    place(i) {
      const row = Math.floor(i / this.cols);
      const k = i % this.cols;
      // --i staggers the glitch so the blurred stops never flicker together
      return { gridRow: row + 1, gridColumn: row % 2 ? k + 1 : this.cols - k, "--i": i };
    },
    async layout() {
      const w = this.$el.clientWidth;
      const cols = w > 860 ? 5 : w > 520 ? 3 : 2;
      if (cols !== this.cols) this.cols = cols;
      await this.$nextTick();
      const box = this.$el.getBoundingClientRect();
      const posts = [...this.$el.querySelectorAll(".posts")];
      if (!posts.length) return;
      // One track height per row: just under the posts of that row's boards.
      const ys = [];
      posts.forEach((p, i) => {
        const row = Math.floor(i / this.cols);
        if (ys[row] == null) ys[row] = p.getBoundingClientRect().bottom - box.top + 3;
      });
      const xa = CURVE + 6;
      const xb = w - CURVE - 6;
      let d = `M ${w - 2} ${ys[0]}`;
      ys.forEach((y, r) => {
        const leftward = r % 2 === 0;
        const next = ys[r + 1];
        if (next == null) d += ` H ${leftward ? 2 : w - 2}`;
        else {
          const ry = (next - y) / 2;
          d += leftward ? ` H ${xa} A ${CURVE} ${ry} 0 0 0 ${xa} ${next}` : ` H ${xb} A ${CURVE} ${ry} 0 0 1 ${xb} ${next}`;
        }
      });
      this.w = w;
      this.h = this.$el.clientHeight;
      this.d = d;
      await this.$nextTick();
      // Where along the track each stop is: the nearest point to its posts.
      const path = this.$refs.path;
      const total = path.getTotalLength();
      const samples = [];
      for (let l = 0; l <= total; l += 4) samples.push([l, path.getPointAtLength(l)]);
      this.stops = posts.map((p) => {
        const r = p.getBoundingClientRect();
        const x = r.left + r.width / 2 - box.left;
        const y = r.bottom - box.top + 3;
        let best = 0;
        let bd = Infinity;
        for (const [l, pt] of samples) {
          const dd = (pt.x - x) ** 2 + (pt.y - y) ** 2;
          if (dd < bd) [best, bd] = [l, dd];
        }
        return best;
      });
      // On first layout the train waits at the start; after a resize it stays
      // at its stop.
      if (!this.moving) this.park(this.stop < 0 ? START : this.stops[this.stop]);
    },
    pointAt(l) {
      const path = this.$refs.path;
      const p = path.getPointAtLength(l);
      const q = path.getPointAtLength(Math.min(path.getTotalLength(), l + 2));
      const r = path.getPointAtLength(Math.max(0, l - 2));
      return { x: p.x, y: p.y, angle: (Math.atan2(q.y - r.y, q.x - r.x) * 180) / Math.PI };
    },
    park(l) {
      this.cur = l;
      this.train = this.pointAt(l);
      this.shown = this.stops.filter((s) => s <= l + 2).length;
    },
    // Ride to stop `to` (-1 = back to the start), easing in and out.
    go(to, ms = HOP_MS) {
      if (this.moving || !this.stops.length) return;
      const from = this.cur;
      const target = to < 0 ? START : this.stops[to];
      if (this.still) {
        this.stop = to;
        return this.park(target);
      }
      const t0 = performance.now();
      const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
      this.moving = true;
      const step = (now) => {
        const t = Math.min(1, (now - t0) / ms);
        this.park(from + (target - from) * ease(t));
        if (t < 1) this.raf = requestAnimationFrame(step);
        else {
          this.moving = false;
          this.stop = to;
        }
      };
      this.raf = requestAnimationFrame(step);
    },
    next() {
      if (this.stop >= this.stages.length - 1) return this.go(-1, 450);
      this.go(this.stop + 1);
    },
    back() {
      if (this.stop >= 0) this.go(this.stop - 1);
    },
    toEnd() {
      this.go(this.stages.length - 1, 1200);
    },
    // A click anywhere on the panel goes to the next stop (the buttons below
    // keep their own jobs).
    clickPanel(e) {
      if (e.target.closest(".xp-controls")) return;
      this.next();
    },
    key(e) {
      if (e.key === "ArrowRight" || e.key === " ") {
        e.preventDefault();
        this.next();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        this.back();
      }
    },
  },
  template: `
    <div class="xp" tabindex="0" aria-label="How a dub is made — press the right arrow for the next stop" @keydown="key" @click="clickPanel">
      <svg class="xp-track" :width="w" :height="h" :viewBox="'0 0 ' + w + ' ' + h" aria-hidden="true">
        <path :d="d" class="rails" />
        <path :d="d" class="rails-gap" />
        <path :d="d" class="sleepers" />
        <path ref="path" :d="d" fill="none" stroke="none" />
      </svg>
      <button class="xp-train" :style="trainStyle" title="Next stop" aria-label="Next stop"><Loco :size="44" /></button>
      <div class="xp-grid" :style="{ gridTemplateColumns: 'repeat(' + cols + ', minmax(0, 1fr))' }">
        <div v-for="(s, i) in stages" :key="s.key" class="halt" :class="{ shown: i < shown, here: i === stop && !moving }" :style="place(i)">
          <div class="board">
            <div class="board-hi">{{ s.hi }}</div>
            <div class="board-en">{{ s.en }}</div>
            <div class="board-alt">{{ s.place }} · {{ s.code }}</div>
          </div>
          <div class="posts" aria-hidden="true"></div>
          <div class="halt-words"><b>{{ i + 1 }}</b> {{ s.words }}</div>
          <code class="halt-tech">{{ s.engine }}</code>
        </div>
      </div>
      <div class="xp-controls">
        <button class="btn sm" :disabled="stop < 0 || moving" @click="back">← Back</button>
        <button class="btn primary sm" :disabled="moving" @click="next">{{ stop < 0 ? 'Start the journey →' : stop < stages.length - 1 ? 'Next stop →' : 'Back to the start' }}</button>
        <button v-if="stop < stages.length - 1" class="btn ghost sm" :disabled="moving" @click="toEnd">Ride to the end</button>
        <span class="xp-count">{{ stop < 0 ? 'Waiting at ' + stages[0].place : 'Stop ' + (stop + 1) + ' of ' + stages.length + ' · ' + stages[stop].place }}</span>
      </div>
    </div>`,
};
