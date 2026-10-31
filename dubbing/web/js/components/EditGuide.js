// "What would you like to change?" — the way into editing, in plain words.
// Each choice says what it does and, when clicked, takes the person to the
// place where it is done and shows them what to do there (Project.js
// handles the "go" event). Above it, a three-step strip shows where they
// are: pick → choose the change → apply.

const TASKS = [
  { key: "moment", icon: "film", title: "A part of the video", sub: "Pick a few seconds, then change the voice, feeling, words or language there" },
  { key: "line", icon: "pen", title: "The words of a line", sub: "Rewrite what is said, or fix the transcript" },
  { key: "voice", icon: "mic", title: "Someone's voice", sub: "A different voice for one person, everywhere in the video" },
  { key: "lips", icon: "lips", title: "Lip sync", sub: "Redo the mouths for the whole video — the audio stays as it is" },
  { key: "names", icon: "book", title: "A name said wrong", sub: "Names, brands or terms to keep exactly right" },
  { key: "ask", icon: "chat", title: "Something else", sub: "Describe it in your own words — we work out the changes" },
  { key: "undo", icon: "loop", title: "Go back", sub: "Undo a change, or return to an earlier version" },
];

export default {
  props: {
    step: { type: Number, default: 1 }, // 1 pick, 2 choose the change, 3 apply
    showTasks: Boolean,
  },
  emits: ["go", "hide", "show"],
  data: () => ({ TASKS }),
  template: `
    <div class="edit-guide">
      <ol class="steps-strip" aria-label="How editing works">
        <li :class="{ on: step === 1, done: step > 1 }"><b>1</b> Pick what to change</li>
        <li :class="{ on: step === 2, done: step > 2 }"><b>2</b> Choose the change</li>
        <li :class="{ on: step === 3 }"><b>3</b> Apply</li>
      </ol>
      <p class="steps-note">Changes wait in a list until you press <b>Apply</b> — nothing is redone before that, and every change can be undone.</p>
      <template v-if="showTasks">
        <div class="eg-head">
          <span>What would you like to change?</span>
          <button class="link" @click="$emit('hide')">Hide</button>
        </div>
        <div class="eg-tasks">
          <button v-for="t in TASKS" :key="t.key" class="eg-task" @click="$emit('go', t.key)">
            <span class="eg-icon"><Icon :name="t.icon" :size="20" /></span>
            <span><b>{{ t.title }}</b><small>{{ t.sub }}</small></span>
          </button>
        </div>
      </template>
      <button v-else class="link eg-show" @click="$emit('show')">What can I change?</button>
    </div>`,
};
