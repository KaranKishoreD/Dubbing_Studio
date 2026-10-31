// The home page's picker in International mode: a real globe (Natural Earth
// 1:50m land and borders, orthographic projection, drawn on a canvas with d3)
// that turns to each language in turn. Each language is pinned everywhere it
// is widely spoken (English in the US, UK, Australia, Nigeria…). Drag to turn,
// scroll or +/− to zoom (Europe is crowded), point at a pin to stop the tour,
// click a pin to dub into that language.
//
// d3 and topojson-client are loaded as globals from /vendor (index.html).

const PINS = {
  en: [[-77, 38.9], [-0.1, 51.5], [149.1, -35.3], [-75.7, 45.4], [7.5, 9.1], [174.8, -41.3]],
  es: [[-3.7, 40.4], [-99.1, 19.4], [-74.1, 4.7], [-58.4, -34.6], [-77.0, -12.0], [-70.7, -33.4]],
  pt: [[-47.9, -15.8], [-9.1, 38.7], [13.2, -8.8], [32.6, -25.9]],
  fr: [[2.35, 48.85], [-73.6, 45.5], [-17.4, 14.7], [15.3, -4.3], [-4.0, 5.3]],
  it: [[12.5, 41.9]], de: [[13.4, 52.5], [16.4, 48.2], [7.4, 46.9]], nl: [[4.9, 52.4]],
  ru: [[37.6, 55.75], [82.9, 55.0], [131.9, 43.1]], uk: [[30.5, 50.45]], pl: [[21.0, 52.2]],
  ar: [[31.2, 30.0], [46.7, 24.7], [-6.8, 34.0], [3.1, 36.7], [44.4, 33.3]],
  sw: [[36.8, -1.3], [39.3, -6.8], [32.6, 0.3]],
  zh: [[116.4, 39.9], [121.5, 31.2], [113.3, 23.1], [121.5, 25.0], [103.8, 1.35]],
  ja: [[139.7, 35.7]], ko: [[127.0, 37.6]], vi: [[105.8, 21.0], [106.7, 10.8]],
  th: [[100.5, 13.75]], id: [[106.8, -6.2], [112.75, -7.25]], fa: [[51.4, 35.7], [69.2, 34.5]],
  tr: [[32.9, 39.9], [29.0, 41.0]], he: [[35.2, 31.8]], el: [[23.7, 38.0]], kk: [[71.4, 51.2]],
  ka: [[44.8, 41.7]], hy: [[44.5, 40.2]], ku: [[44.0, 36.2]], sv: [[18.0, 59.3]], no: [[10.75, 59.9]],
  da: [[12.6, 55.7]], fi: [[24.9, 60.2]], is: [[-21.9, 64.1]], cs: [[14.4, 50.1]], sk: [[17.1, 48.15]],
  hu: [[19.0, 47.5]], ro: [[26.1, 44.4]], bg: [[23.3, 42.7]], sr: [[20.5, 44.8]], sl: [[14.5, 46.0]],
  sq: [[19.8, 41.3]], ca: [[2.2, 41.4]], eu: [[-2.9, 43.3]], cy: [[-3.2, 51.5]], et: [[24.75, 59.4]],
  lv: [[24.1, 56.9]], lt: [[25.3, 54.7]], lb: [[6.1, 49.6]],
};
const MIN_ZOOM = 1;
const MAX_ZOOM = 8;
let WORLD = null; // {coarse, fine: {land, borders}}, loaded once per page

async function loadWorld() {
  if (WORLD) return WORLD;
  const get = (f) => fetch(`/vendor/${f}`).then((r) => r.json());
  const [l1, c1, l5, c5] = await Promise.all(
    ["land-110m.json", "countries-110m.json", "land-50m.json", "countries-50m.json"].map(get),
  );
  const pack = (land, countries) => ({
    land: topojson.feature(land, land.objects.land),
    borders: topojson.mesh(countries, countries.objects.countries, (a, b) => a !== b),
  });
  // 1:110m while the whole globe turns (light), 1:50m once zoomed in (sharp)
  WORLD = { coarse: pack(l1, c1), fine: pack(l5, c5) };
  return WORLD;
}

export default {
  props: {
    langs: { type: Array, required: true }, // [{code, en, native}]
    active: { type: Number, default: 0 },
  },
  emits: ["pick", "active"],
  data: () => ({ zoom: 1, ready: false, still: matchMedia("(prefers-reduced-motion: reduce)").matches }),
  watch: {
    // A new language in the tour: turn to its first pin (unless zoomed in).
    active() {
      const at = (PINS[this.langs[this.active]?.code] || [])[0];
      if (at && this.zoom < 1.5) this.target = [-at[0], -Math.max(-35, Math.min(50, at[1] * 0.75))];
    },
  },
  async mounted() {
    this.rot = [10, -20]; // [lambda, phi] rotation
    this.target = null;
    this.hover = null;
    this.drag = null;
    this.world = await loadWorld();
    this.ready = true;
    this.$nextTick(() => {
      this.resize();
      this.ro = new ResizeObserver(() => this.resize());
      this.ro.observe(this.$refs.wrap);
      const tick = () => {
        this.step();
        this.draw();
        this.raf = requestAnimationFrame(tick);
      };
      this.raf = requestAnimationFrame(tick);
    });
    this.timer = setInterval(() => {
      if (this.hover == null && !this.drag && this.zoom < 1.5 && this.langs.length)
        this.$emit("active", (this.active + 1) % this.langs.length);
    }, 2800);
  },
  unmounted() {
    cancelAnimationFrame(this.raf);
    clearInterval(this.timer);
    this.ro?.disconnect();
  },
  methods: {
    resize() {
      const c = this.$refs.canvas;
      if (!c) return;
      const w = this.$refs.wrap.clientWidth;
      const dpr = window.devicePixelRatio || 1;
      c.width = c.height = Math.round(w * dpr);
      c.style.width = c.style.height = w + "px";
      this.size = w;
      this.dpr = dpr;
    },
    projection() {
      const r = this.size * 0.42;
      return d3
        .geoOrthographic()
        .scale(r * this.zoom)
        .translate([this.size / 2, this.size / 2])
        .rotate(this.rot)
        .clipAngle(90)
        .precision(0.5);
    },
    step() {
      if (this.drag) return;
      if (this.target) {
        const d = ((this.target[0] - this.rot[0] + 540) % 360) - 180;
        this.rot = [this.rot[0] + d * 0.07, this.rot[1] + (this.target[1] - this.rot[1]) * 0.07];
        if (Math.abs(d) < 0.15 && Math.abs(this.target[1] - this.rot[1]) < 0.15) this.target = null;
      } else if (this.hover == null && this.zoom < 1.5 && !this.still) {
        this.rot = [this.rot[0] + 0.05, this.rot[1]];
      }
    },
    // Visible pins: [{lang index, code, x, y}].
    visiblePins(p) {
      const centre = [-this.rot[0], -this.rot[1]];
      const out = [];
      this.langs.forEach((l, i) => {
        for (const ll of PINS[l.code] || []) {
          if (d3.geoDistance(ll, centre) > Math.PI / 2 - 0.05) continue;
          const [x, y] = p(ll);
          out.push({ i, code: l.code, x, y });
        }
      });
      return out;
    },
    draw() {
      const c = this.$refs.canvas;
      if (!c || !this.world) return;
      const ctx = c.getContext("2d");
      const css = getComputedStyle(this.$el);
      const v = (n) => css.getPropertyValue(n).trim();
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      ctx.clearRect(0, 0, this.size, this.size);
      const p = this.projection();
      const path = d3.geoPath(p, ctx);
      const [cx, cy] = p.translate();
      const r = p.scale();
      // ocean, lit from the top left
      const g = ctx.createRadialGradient(cx - r * 0.35, cy - r * 0.4, r * 0.1, cx, cy, r);
      g.addColorStop(0, v("--globe-sea-hi"));
      g.addColorStop(1, v("--globe-sea"));
      ctx.beginPath();
      path({ type: "Sphere" });
      ctx.fillStyle = g;
      ctx.fill();
      // latitude / longitude lines
      ctx.beginPath();
      path(d3.geoGraticule10());
      ctx.strokeStyle = v("--globe-grid");
      ctx.lineWidth = 0.6;
      ctx.stroke();
      const map = this.zoom >= 1.6 ? this.world.fine : this.world.coarse;
      // land with a coastline
      ctx.beginPath();
      path(map.land);
      ctx.fillStyle = v("--globe-land");
      ctx.fill();
      ctx.strokeStyle = v("--globe-coast");
      ctx.lineWidth = 1;
      ctx.stroke();
      // borders
      ctx.beginPath();
      path(map.borders);
      ctx.strokeStyle = v("--globe-border");
      ctx.lineWidth = 0.5;
      ctx.stroke();
      // rim
      ctx.beginPath();
      path({ type: "Sphere" });
      ctx.strokeStyle = v("--globe-rim");
      ctx.lineWidth = 1.5;
      ctx.stroke();
      // pins
      const pins = (this.pins = this.visiblePins(p));
      for (const pin of pins) {
        const on = pin.i === this.active;
        ctx.beginPath();
        ctx.arc(pin.x, pin.y, on ? 6 : 4, 0, 2 * Math.PI);
        ctx.fillStyle = on ? "#ffffff" : v("--accent");
        ctx.fill();
        ctx.lineWidth = on ? 3 : 1.5;
        ctx.strokeStyle = on ? v("--accent") : "#ffffff";
        ctx.stroke();
      }
      // the active language's name at each of its visible pins
      const l = this.langs[this.active];
      if (l) {
        ctx.textAlign = "center";
        for (const pin of pins.filter((q) => q.i === this.active)) {
          ctx.font = "700 15px Mukta, 'Noto Sans', sans-serif";
          ctx.lineJoin = "round";
          ctx.lineWidth = 4;
          ctx.strokeStyle = "rgba(4, 30, 38, .85)";
          ctx.strokeText(l.native, pin.x, pin.y - 12);
          ctx.fillStyle = "#ffffff";
          ctx.fillText(l.native, pin.x, pin.y - 12);
        }
      }
    },
    pinAt(e) {
      const b = this.$refs.canvas.getBoundingClientRect();
      const x = e.clientX - b.left;
      const y = e.clientY - b.top;
      let best = null;
      let bd = 12 * 12;
      for (const pin of this.pins || []) {
        const d = (pin.x - x) ** 2 + (pin.y - y) ** 2;
        if (d < bd) [best, bd] = [pin, d];
      }
      return best;
    },
    down(e) {
      if (e.target.closest("button")) return;
      this.drag = { x: e.clientX, y: e.clientY, rot: [...this.rot], moved: false };
      this.target = null;
      e.currentTarget.setPointerCapture?.(e.pointerId);
    },
    move(e) {
      if (this.drag) {
        const k = 0.35 / this.zoom;
        const dx = e.clientX - this.drag.x;
        const dy = e.clientY - this.drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) this.drag.moved = true;
        this.rot = [this.drag.rot[0] + dx * k, Math.max(-80, Math.min(80, this.drag.rot[1] - dy * k))];
        return;
      }
      const pin = this.pinAt(e);
      this.$refs.canvas.style.cursor = pin ? "pointer" : "grab";
      if ((pin && pin.i) !== this.hover) {
        this.hover = pin ? pin.i : null;
        if (pin) this.$emit("active", pin.i);
      }
    },
    up(e) {
      const was = this.drag;
      this.drag = null;
      if (was && !was.moved) {
        const pin = this.pinAt(e);
        if (pin) this.$emit("pick", this.langs[pin.i]);
      }
    },
    leave() {
      this.drag = null;
      this.hover = null;
    },
    wheel(e) {
      e.preventDefault();
      this.setZoom(this.zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15));
    },
    setZoom(z) {
      this.zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z));
      this.target = null;
    },
    // Zoom straight into Europe, where most pins are.
    europe() {
      this.target = [-15, -50];
      this.zoom = 3.2;
    },
  },
  template: `
    <div class="globe" ref="wrap" @pointerdown="down" @pointermove="move" @pointerup="up" @pointerleave="leave" @wheel="wheel">
      <canvas ref="canvas" role="img" aria-label="Globe with a pin for each language — click a pin to dub into it"></canvas>
      <div v-if="!ready" class="globe-loading">loading the map…</div>
      <div class="globe-zoom">
        <button class="btn sm" @click="setZoom(zoom * 1.4)" title="Zoom in">+</button>
        <button class="btn sm" @click="setZoom(zoom / 1.4)" title="Zoom out">−</button>
        <button class="btn sm" @click="europe" title="Zoom into Europe">Europe</button>
        <button class="btn sm" @click="setZoom(1)" title="Whole globe">World</button>
      </div>
      <div class="bhint">drag to turn · scroll to zoom · click a pin</div>
    </div>`,
};
