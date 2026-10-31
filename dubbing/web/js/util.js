// Small helpers used across the page.

export function fmtTime(sec) {
  if (sec == null || isNaN(sec)) return "–";
  sec = Math.max(0, Math.floor(sec));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = String(sec % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
}

export function fmtSize(bytes) {
  if (bytes > 1e6) return (bytes / 1e6).toFixed(1) + " MB";
  return Math.max(1, Math.round(bytes / 1e3)) + " KB";
}

export function ago(epochSeconds) {
  const s = Date.now() / 1000 - epochSeconds;
  if (s < 90) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(epochSeconds * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

// One colour per speaker, the same everywhere on the page.
// marigold, peacock, mehendi, jamun, haldi, gulabi, neel, kumkum, …
const SPEAKER_COLORS = ["#ff9a1f", "#25c0b0", "#7cc44a", "#b07cf2", "#f5c84c", "#f472b6", "#6d7cff", "#ec3b72", "#2fb5e8", "#e0a370", "#a3e635", "#ff6f3c"];
export const speakerColor = (n) => SPEAKER_COLORS[n % SPEAKER_COLORS.length];
export const speakerName = (n) => `Speaker ${n + 1}`;

// "Hindi (हिन्दी)" → { en: "Hindi", native: "हिन्दी" }
export function splitLangLabel(label) {
  const m = label.match(/^(.*?)\s*\((.*)\)$/);
  return m ? { en: m[1], native: m[2] } : { en: label, native: label };
}

export function youtubeId(url) {
  const m = (url || "").match(/(?:v=|youtu\.be\/|shorts\/|embed\/)([A-Za-z0-9_-]{11})/);
  return m ? m[1] : null;
}

// The server writes short status notes in a little Markdown (bold, bullet
// lists). This turns exactly that into safe HTML.
export function miniMarkdown(text) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|\s)_([^_]+)_(?=\s|$)/g, "$1<i>$2</i>");
  const out = [];
  let list = [];
  const flush = () => {
    if (list.length) out.push("<ul>" + list.map((li) => `<li>${inline(li)}</li>`).join("") + "</ul>");
    list = [];
  };
  for (const raw of (text || "").split("\n")) {
    const line = raw.trim();
    if (line.startsWith("- ")) list.push(line.slice(2));
    else {
      flush();
      if (line) out.push(`<p>${inline(line.replace(/^#+\s*/, ""))}</p>`);
    }
  }
  flush();
  return out.join("");
}

// Remaining time in words: "under a minute", "about 6 min", "about 1 h 10 min".
export function fmtLeft(sec) {
  if (sec == null) return "";
  if (sec < 50) return "under a minute";
  const min = Math.round(sec / 60);
  if (min < 60) return `about ${min} min`;
  return `about ${Math.floor(min / 60)} h ${min % 60} min`;
}

// The clock time a job should be done, e.g. "7:42 PM".
export const finishAt = (sec) =>
  new Date(Date.now() + sec * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

// What each pipeline stage is doing, in plain words, for progress displays.
export const STAGE_WORDS = {
  download: "Fetching the video",
  separate: "Separating the voices from the music",
  transcribe: "Listening to what is said",
  diarize: "Working out who says what",
  refine: "Tidying up the transcript",
  translate: "Translating, line by line",
  synthesize: "Giving everyone a voice",
  align: "Fitting each line to the speaker's timing",
  lipsync: "Moving lips to the new words",
  assemble: "Putting it all together",
};

// The pipeline as a train journey, the "Dub Express", running north to south
// along the Himsagar Express route from Jammu Tawi to Kanyakumari. Each stage
// is a stop: its yellow board says what the step does (Hindi above, English
// below); the real station and its code are only the small print.
export const STATIONS = {
  download: { hi: "वीडियो लाना", en: "FETCH VIDEO", place: "Jammu Tawi", code: "JAT" },
  separate: { hi: "आवाज़ अलग", en: "SPLIT VOICE & MUSIC", place: "Ludhiana Jn.", code: "LDH" },
  transcribe: { hi: "सुनना", en: "LISTEN", place: "New Delhi", code: "NDLS" },
  diarize: { hi: "कौन बोला", en: "WHO SPEAKS", place: "Agra Cantt", code: "AGC" },
  refine: { hi: "साफ़ करना", en: "TIDY THE TEXT", place: "Bhopal Jn.", code: "BPL" },
  translate: { hi: "अनुवाद", en: "TRANSLATE", place: "Nagpur Jn.", code: "NGP" },
  synthesize: { hi: "नई आवाज़", en: "NEW VOICES", place: "Vijayawada Jn.", code: "BZA" },
  align: { hi: "तालमेल", en: "FIT TIMING", place: "Coimbatore Jn.", code: "CBE" },
  lipsync: { hi: "होंठ मिलाना", en: "LIP SYNC", place: "Thiruvananthapuram", code: "TVC" },
  assemble: { hi: "जोड़ना", en: "FINAL MIX", place: "Kanyakumari", code: "CAPE" },
};

// What runs under the hood at each stage, for the curious.
export const ENGINE = {
  download: "yt-dlp · ffmpeg",
  separate: "Demucs htdemucs",
  transcribe: "faster-whisper large-v3",
  diarize: "pyannote",
  refine: "LLM clean-up",
  translate: "LLM dubbing director",
  synthesize: "Bodhan · Sarvam TTS",
  align: "Seed-VC clone + time-fit",
  lipsync: "LatentSync 1.6",
  assemble: "ffmpeg mix",
};

// The first letter of a language in its own script, for small round badges:
// "Hindi (हिन्दी)" → "हि", "English" → "En".
export function glyph(label) {
  const { native } = splitLangLabel(label || "");
  const chars = [...native];
  if (!chars.length) return "?";
  if (/^[A-Za-z]/.test(native)) return chars.slice(0, 2).join("");
  // Keep a base letter together with the vowel signs and marks that follow it.
  const marks = /[ऀ-ःऺ-ॏॕ-ॗॢॣঁ-ঃ়-্ৗਁ-ਃ਼-੍ઁ-ઃ઼-્ଁ-ଃ଼-୍ஂா-்ఀ-ఄా-ౖಁ-ಃ಼-ೖഀ-ഃാ-്ൗً-ٟ]/;
  let out = chars[0];
  for (const c of chars.slice(1)) {
    if (marks.test(c)) out += c;
    else break;
  }
  return out;
}
