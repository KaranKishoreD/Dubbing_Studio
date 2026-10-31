// The home page's language picker, two ways round on one card (flip it):
//
//   Lotus: a 3D lotus that breathes open and closed on its lily pad, with
//      every language orbiting it. The orbit turns on its own, can be dragged
//      to spin, and the language passing in front is the "active" one (the
//      headline follows it). The lotus tilts towards the mouse.
//   Map: India drawn in kolam dots. Each language has a pin where it is
//      spoken; pointing at one lights up its region, dot by dot from the pin.
//
// Clicking a language anywhere emits "pick" (start a dub into it).

const TAU = Math.PI * 2;

// A rough outline of India (longitude, latitude), enough for a dot map.
const INDIA = [
  [68.75, 23.95], [69.2, 22.8], [68.95, 22.3], [69.8, 21.5], [70.8, 20.75], [72.1, 21.2], [72.6, 22.2], [72.85, 21.0],
  [72.8, 19.9], [72.85, 18.9], [73.1, 17.6], [73.5, 16.0], [73.9, 15.2], [74.4, 14.0], [74.8, 12.8], [75.3, 11.8],
  [75.8, 11.0], [76.25, 9.95], [76.6, 8.9], [77.55, 8.05], [78.2, 8.85], [78.6, 9.3], [79.3, 10.3], [79.85, 10.8],
  [79.85, 12.0], [80.3, 13.3], [80.2, 15.0], [80.9, 15.8], [81.6, 16.3], [82.3, 16.6], [83.3, 17.6], [84.2, 18.3],
  [85.1, 19.4], [86.4, 19.95], [86.9, 20.8], [87.1, 21.5], [88.2, 21.6], [88.9, 21.6], [89.05, 22.1], [88.75, 23.0],
  [88.6, 24.0], [88.1, 24.5], [88.45, 25.15], [89.0, 25.3], [89.85, 25.3], [90.5, 25.15], [91.5, 25.15], [92.2, 25.0],
  [92.0, 24.5], [91.3, 24.1], [91.15, 23.6], [91.6, 22.95], [92.0, 23.3], [92.3, 22.6], [92.6, 21.95], [93.1, 22.1],
  [93.4, 23.0], [93.4, 23.9], [94.2, 23.9], [94.6, 24.7], [94.8, 25.5], [95.2, 26.0], [95.5, 26.7], [96.2, 27.3],
  [97.1, 27.5], [97.35, 28.2], [96.6, 28.6], [96.0, 29.2], [95.3, 29.0], [94.3, 29.2], [93.2, 28.6], [92.5, 27.9],
  [91.65, 27.8], [91.65, 26.85], [90.0, 26.8], [89.1, 26.85], [88.9, 27.3], [88.85, 28.0], [88.15, 27.9], [88.05, 26.5],
  [87.0, 26.4], [86.0, 26.6], [85.0, 26.9], [84.1, 27.4], [83.3, 27.35], [82.0, 27.9], [81.0, 28.4], [80.05, 28.85],
  [80.25, 30.0], [79.2, 31.0], [78.7, 31.6], [78.8, 32.5], [79.3, 33.2], [78.9, 34.3], [80.2, 35.2], [79.8, 35.6],
  [78.0, 35.5], [77.8, 35.9], [76.5, 35.9], [75.6, 36.8], [74.9, 37.05], [74.0, 36.9], [73.2, 36.6], [72.6, 36.0],
  [73.6, 35.0], [73.9, 34.3], [73.6, 33.6], [74.0, 33.0], [74.6, 32.6], [75.0, 32.1], [74.6, 31.0], [73.9, 30.2],
  [73.4, 29.9], [72.9, 29.0], [71.9, 27.9], [70.7, 27.8], [70.0, 27.0], [69.6, 26.6], [70.15, 25.8], [70.65, 25.0],
  [71.1, 24.4], [70.0, 24.2], [69.0, 24.3],
];

// Where each language is pinned. English and Sanskrit are spoken (and
// studied) everywhere, so they have no region of their own: English floats
// at sea, Sanskrit sits at Varanasi, and either lights up the whole map.
const PINS = {
  hi: [78.0, 23.6], ur: [80.95, 26.85], pa: [75.5, 30.9], gu: [71.8, 22.6], mr: [75.0, 19.2], bn: [88.1, 23.0],
  or: [84.6, 20.4], as: [92.9, 26.4], te: [79.5, 16.8], kn: [76.2, 14.2], ml: [76.4, 10.2], ta: [78.7, 11.0],
  ne: [88.4, 27.1], sa: [83.0, 25.3], mai: [86.0, 26.1], kok: [74.0, 15.4], doi: [74.9, 32.7], ks: [74.8, 34.1],
  sd: [70.1, 23.2], mni: [93.9, 24.8], brx: [90.6, 26.5], sat: [87.0, 24.2], en: [70.4, 12.6],
};
const EVERYWHERE = new Set(["en", "sa"]);

// Where each name sits relative to its pin [dx, dy, text-anchor], so that all
// 23 can be read at once. Most sit to the right of the pin; in crowded spots
// (the east, the north-west) they move above, below or out to sea, with a
// dotted leader line back to the pin.
const LABELS = {
  ks: [7, -6], doi: [7, -6], pa: [7, -6], hi: [7, -9], ur: [7, -9], sa: [0, 12, "middle"],
  mai: [-14, -22, "middle"], ne: [-8, -21, "middle"], sat: [45, 18], bn: [34, 22], brx: [0, -21, "middle"],
  as: [7, -6], mni: [7, -4], gu: [7, -9], sd: [0, -21, "middle"], mr: [7, -9], te: [7, -9], kn: [7, -9],
  or: [7, -9], ta: [7, -9], ml: [-7, -9, "end"], kok: [-7, -9, "end"], en: [0, -21, "middle"],
};

// Equirectangular, squeezed a little east–west for India's latitude.
const px = ([lon, lat]) => [(lon - 67.5) * 9.3, (37.6 - lat) * 10];

function inside([x, y], poly) {
  let hit = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
  }
  return hit;
}

// The dots of the map, each belonging to the nearest pin (its "region").
const DOTS = (() => {
  const out = [];
  const regional = Object.entries(PINS).filter(([code]) => !EVERYWHERE.has(code));
  for (let lat = 37.2; lat > 7.8; lat -= 0.62) {
    for (let lon = 68.2; lon < 97.6; lon += 0.62) {
      if (!inside([lon, lat], INDIA)) continue;
      let best = null;
      let bd = Infinity;
      for (const [code, [plon, plat]] of regional) {
        const d = Math.hypot(lon - plon, lat - plat);
        if (d < bd) [best, bd] = [code, d];
      }
      const [x, y] = px([lon, lat]);
      out.push({ x, y, lang: best, d: bd });
    }
  }
  return out;
})();

const LOTUS = [
  // petals per ring, lift from the water (deg), offset (deg), size
  { n: 11, lift: 16, off: 0, w: 56, h: 128, tone: "outer" },
  { n: 10, lift: 40, off: 18, w: 50, h: 112, tone: "mid" },
  { n: 7, lift: 66, off: 9, w: 42, h: 88, tone: "inner" },
];

export default {
  props: {
    langs: { type: Array, required: true }, // [{code, en, native, glyph}]
    active: { type: Number, default: 0 },
  },
  emits: ["pick", "active"],
  data() {
    let view = "lotus";
    try {
      view = localStorage.getItem("stage-view") || "lotus";
    } catch {
      /* the default */
    }
    return {
      view,
      angle: 0, // orbit rotation (radians)
      spin: 0.2, // radians per second when left alone
      hovering: null, // index of the language under the pointer
      drag: null, // {x, angle, t, v}
      tiltX: 0,
      tiltY: 0,
      ripple: 0, // bumps to restart the ripple animation
      raf: 0,
      last: 0,
      cycle: 0,
      slotOf: null, // language index → place on the orbit (widest names on the outer ring)
      LOTUS,
      DOTS,
      still: matchMedia("(prefers-reduced-motion: reduce)").matches,
    };
  },
  computed: {
    pins() {
      return this.langs.map((l, i) => {
        const [dx, dy, anchor = "start"] = LABELS[l.code] || [7, -9];
        // a leader line when the name is far from the pin's head
        const far = Math.hypot(dx, dy + 12) > 26;
        return { ...l, i, at: px(PINS[l.code] || [72, 10]), label: { dx, dy, anchor, far } };
      });
    },
    // Places on the orbit: an even number, so the two rings always alternate
    // (with an odd number of languages one place stays empty).
    slots() {
      return Math.max(2, this.langs.length + (this.langs.length % 2));
    },
    // The orbit place of each language, and the language at each place.
    places() {
      return this.slotOf || this.langs.map((_, i) => i);
    },
    langAt() {
      const out = Array(this.slots).fill(null);
      this.places.forEach((slot, i) => (out[slot] = i));
      return out;
    },
    activeCode() {
      return this.langs[this.active]?.code;
    },
    everywhere() {
      return EVERYWHERE.has(this.activeCode);
    },

    tilt() {
      return { "--tx": this.tiltX + "deg", "--ty": this.tiltY + "deg" };
    },
  },
  watch: {
    active() {
      this.ripple++;
    },
  },
  mounted() {
    const tick = (t) => {
      const dt = Math.min(0.05, (t - (this.last || t)) / 1000);
      this.last = t;
      if (this.view === "lotus" && !this.drag && this.hovering == null && !this.still) {
        this.angle += this.spin * dt;
        // after a fling, ease back to the resting speed
        this.spin += (0.2 - this.spin) * Math.min(1, dt * 1.5);
        this.front();
      }
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
    document.fonts.ready.then(() => this.$nextTick(this.arrange));
    // On the map nothing turns, so the active language steps along by itself.
    this.cycle = setInterval(() => {
      if (this.view === "map" && this.hovering == null && this.langs.length) this.$emit("active", (this.active + 1) % this.langs.length);
    }, 2400);
  },
  unmounted() {
    cancelAnimationFrame(this.raf);
    clearInterval(this.cycle);
  },
  methods: {
    // Position of language i on the orbit: an ellipse seen at the lotus's tilt.
    // depth is +1 at the front (closest), -1 at the back.
    orbit(i) {
      const slot = this.places[i];
      const a = this.angle + (TAU * slot) / this.slots;
      const depth = Math.sin(a);
      const s = 0.8 + 0.18 * (depth + 1) / 2;
      // every other place is on a smaller inner ring, so neighbours never collide
      const k = slot % 2 ? 0.72 : 1;
      return {
        left: 50 + 45 * k * Math.cos(a) + "%",
        top: 53 + 39 * k * depth + "%",
        transform: `translate(-50%, -50%) scale(${s.toFixed(3)})`,
        opacity: (0.78 + 0.22 * (depth + 1) / 2).toFixed(2),
        zIndex: depth > 0 ? 20 + Math.round(depth * 10) : 1,
      };
    },
    // Names differ a lot in width (compare बड़ो and ᱥᱟᱱᱛᱟᱲᱤ). Once the fonts
    // are in, the widest half goes on the roomier outer ring (even places),
    // the rest on the inner ring, each keeping the language list's order.
    arrange() {
      const els = [...this.$el.querySelectorAll(".orb .orb-native")];
      if (els.length !== this.langs.length) return;
      const byWidth = els.map((e, i) => [e.offsetWidth, i]).sort((a, b) => b[0] - a[0]);
      const outer = new Set(byWidth.slice(0, this.slots / 2).map(([, i]) => i));
      const slotOf = [];
      let even = 0;
      let odd = 1;
      this.langs.forEach((_, i) => {
        if (outer.has(i)) slotOf[i] = (even += 2) - 2;
        else slotOf[i] = (odd += 2) - 2;
      });
      this.slotOf = slotOf;
    },
    // The language passing in front becomes the active one.
    front() {
      if (!this.langs.length) return;
      // sin(a) is largest at a = π/2
      const k = Math.round(((Math.PI / 2 - this.angle) / TAU) * this.slots);
      const i = this.langAt[((k % this.slots) + this.slots) % this.slots];
      // the empty place at the front keeps the language that was there
      if (i != null && i !== this.active) this.$emit("active", i);
    },
    hover(i) {
      this.hovering = i;
      if (i != null) this.$emit("active", i);
    },
    pick(l) {
      this.ripple++;
      this.$emit("pick", l);
    },
    flip(v) {
      this.view = v;
      try {
        localStorage.setItem("stage-view", v);
      } catch {
        /* not remembered */
      }
    },
    // The lotus leans towards the pointer; dragging spins the orbit.
    move(e) {
      const r = this.$el.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width - 0.5;
      const y = (e.clientY - r.top) / r.height - 0.5;
      this.tiltX = -y * 16;
      this.tiltY = x * 20;
      if (this.drag) {
        const now = performance.now();
        const da = (e.clientX - this.drag.x) * 0.012;
        this.angle = this.drag.angle - da;
        this.drag.v = (-(e.clientX - this.drag.lx) * 0.012) / Math.max(0.016, (now - this.drag.t) / 1000);
        this.drag.lx = e.clientX;
        this.drag.t = now;
        this.front();
      }
    },
    down(e) {
      if (this.view !== "lotus" || e.target.closest(".orb, .bstage-switch")) return;
      this.drag = { x: e.clientX, lx: e.clientX, angle: this.angle, t: performance.now(), v: 0 };
      e.currentTarget.setPointerCapture?.(e.pointerId);
    },
    up() {
      if (!this.drag) return;
      this.spin = Math.max(-6, Math.min(6, this.drag.v || 0.2));
      this.drag = null;
    },
    leave() {
      this.tiltX = this.tiltY = 0;
      this.hovering = null;
      this.up();
    },
    // Arrow keys turn the orbit one place at a time; Enter picks.
    key(e) {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        const step = e.key === "ArrowRight" ? 1 : -1;
        let slot = this.places[this.active] ?? 0;
        do slot = (slot + step + this.slots) % this.slots;
        while (this.langAt[slot] == null);
        this.$emit("active", this.langAt[slot]);
        this.angle = Math.PI / 2 - (TAU * slot) / this.slots;
        this.spin = 0;
      } else if (e.key === "Enter" && this.langs[this.active]) this.pick(this.langs[this.active]);
    },
  },
  template: `
    <div class="bstage" :class="['as-' + view, { dragging: drag }]" :style="tilt"
         @pointermove="move" @pointerdown="down" @pointerup="up" @pointerleave="leave"
         tabindex="0" role="listbox" :aria-activedescendant="'bl-' + activeCode" aria-label="Languages — arrow keys to turn, Enter to dub"
         @keydown="key">
      <div class="bstage-switch" role="tablist">
        <button role="tab" :aria-selected="view === 'lotus'" :class="{ on: view === 'lotus' }" @click="flip('lotus')"><Icon name="lotus" :size="14" /> Lotus</button>
        <button role="tab" :aria-selected="view === 'map'" :class="{ on: view === 'map' }" @click="flip('map')"><Icon name="map" :size="14" /> Map</button>
      </div>

      <div class="bcard">
        <!-- Face 1: the lotus and its orbit -->
        <div class="bface front">
          <div class="pond" aria-hidden="true">
            <div class="water">
              <i class="ring-r" v-for="k in 3" :key="ripple + '-' + k" :style="{ animationDelay: (k - 1) * 0.5 + 's' }"></i>
              <div class="pad"></div>
              <div class="lotus">
                <div v-for="(ring, r) in LOTUS" :key="r" class="ring" :class="ring.tone" :style="{ '--d': r * 0.6 + 's' }">
                  <i v-for="k in ring.n" :key="k" class="petal"
                     :style="{ '--rot': (k * 360 / ring.n + ring.off) + 'deg', '--lift': ring.lift + 'deg', '--w': ring.w + 'px', '--h': ring.h + 'px', '--k': k }"></i>
                </div>
                <div class="pod"><b v-for="k in 7" :key="k"></b></div>
              </div>
            </div>
          </div>
          <button v-for="(l, i) in langs" :key="l.code" :id="'bl-' + l.code" class="orb" :class="{ on: i === active }" :style="orbit(i)"
                  role="option" :aria-selected="i === active" :lang="l.code" :title="'Dub into ' + l.en"
                  @mouseenter="hover(i)" @mouseleave="hover(null)" @click="pick(l)">
            <span class="orb-native">{{ l.native }}</span><span class="orb-en">{{ l.en }}</span>
          </button>
          <div class="bhint">drag to spin · click a language</div>
        </div>

        <!-- Face 2: the map -->
        <div class="bface back">
          <svg class="imap" viewBox="0 0 285 300" aria-hidden="false">
            <g class="tilt">
              <circle v-for="(d, k) in DOTS" :key="k" :cx="d.x" :cy="d.y" r="1.9" class="mdot"
                      :class="{ on: d.lang === activeCode, all: everywhere }"
                      :style="{ transitionDelay: (d.lang === activeCode ? d.d * 45 : 0) + 'ms', '--c': 'var(--lc-' + d.lang + ')' }" />
              <g v-for="p in pins" :key="p.code" class="pin" :class="{ on: p.i === active }" :style="{ '--c': 'var(--lc-' + p.code + ')' }" :transform="'translate(' + p.at[0] + ',' + p.at[1] + ')'"
                 @mouseenter="hover(p.i)" @mouseleave="hover(null)" @click="pick(p)" role="button" :aria-label="'Dub into ' + p.en">
                <circle r="11" class="pin-hit" />
                <circle r="9" class="pin-pulse" />
                <path d="M0 0 C-4 -6 -6 -9 -6 -12 A6 6 0 1 1 6 -12 C6 -9 4 -6 0 0Z" class="pin-drop" />
                <circle r="2.2" cy="-12" class="pin-eye" />
                <title>{{ p.en }}</title>
              </g>
              <!-- every name, drawn after the pins so none hides under a pin -->
              <g v-for="p in pins" :key="'n' + p.code" :transform="'translate(' + p.at[0] + ',' + p.at[1] + ')'" :style="{ '--c': 'var(--lc-' + p.code + ')' }"
                 class="pin" @mouseenter="hover(p.i)" @mouseleave="hover(null)" @click="pick(p)">
                <line v-if="p.label.far" class="leader" x1="0" y1="-12" :x2="p.label.dx" :y2="p.label.dy - 3" />
                <text class="pin-label" :class="{ on: p.i === active }" :x="p.label.dx" :y="p.label.dy" :text-anchor="p.label.anchor">{{ p.native }}</text>
              </g>
            </g>
          </svg>
          <div class="bhint">point at a region · click a pin</div>
        </div>
      </div>
    </div>`,
};
