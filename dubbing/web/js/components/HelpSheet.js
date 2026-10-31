// The help sheet (press ? anywhere, or the ? button): how to do the common
// things, and the keyboard shortcuts.

export default {
  emits: ["close"],
  mounted() {
    this.onKey = (e) => e.key === "Escape" && this.$emit("close");
    window.addEventListener("keydown", this.onKey);
  },
  unmounted() {
    window.removeEventListener("keydown", this.onKey);
  },
  template: `
    <div class="backdrop" @click.self="$emit('close')">
      <div class="dialog help">
        <div class="dialog-head">
          <h2>How do I…?</h2>
          <span style="flex:1"></span>
          <button class="btn ghost icon" aria-label="Close" @click="$emit('close')">✕</button>
        </div>
        <div class="dialog-body">
          <div class="howto">
            <div><span></span><div><b>Dub a video</b><p>Paste a video link (YouTube, Instagram, Vimeo, X…) or drop a file on the home page, pick a language, press Start. Short clips are ready fastest.</p></div></div>
            <div><span></span><div><b>Change one part</b><p>Drag across the film strip under the video (or press <kbd>[</kbd> and <kbd>]</kbd> while it plays). Pick what should change: voice, who's speaking, feeling, words, language.</p></div></div>
            <div><span></span><div><b>Give someone another voice</b><p>Open the Voices tab, choose a voice, press ▶ to hear it, then add it to your changes.</p></div></div>
            <div><span></span><div><b>Fix a word</b><p>In the Lines tab, hover a line: ✎ Dub changes the new words, ✎ Transcript fixes what was heard.</p></div></div>
            <div><span></span><div><b>Apply</b><p>Changes wait in a list at the top until you press Apply. It tells you which lines will be redone and how long it takes.</p></div></div>
            <div><span>↶</span><div><b>Go back</b><p>↶ Undo reverses the last change. The Versions tab has every version — play any of them, or go back to it.</p></div></div>
            <div><span></span><div><b>Go faster</b><p>Untick Lip sync for a quick draft. Add lip sync later with one button.</p></div></div>
          </div>
          <div class="label" style="margin:18px 0 8px">Keyboard</div>
          <div class="keys-grid">
            <span><kbd>Space</kbd> play / pause</span>
            <span><kbd>O</kbd> original ↔ dub</span>
            <span><kbd>C</kbd> side-by-side compare</span>
            <span><kbd>[</kbd> <kbd>]</kbd> mark start / end</span>
            <span><kbd>L</kbd> loop the marked moment</span>
            <span><kbd>Esc</kbd> clear the mark</span>
            <span><kbd>?</kbd> this help</span>
          </div>
        </div>
      </div>
    </div>`,
};
