// A clean player to share: the dub in every language it exists in, the
// original, and Compare — without the editing tools. Link: #/watch/<id>/<lang>.

import { api } from "../api.js";
import { go } from "../store.js";
import Player from "../components/Player.js";
import CommentsCard from "../components/CommentsCard.js";

export default {
  components: { Player, CommentsCard },
  props: { id: { type: String, required: true }, lang: { type: String, default: null } },
  data: () => ({ project: null, error: "", now: 0, comments: [] }),
  watch: { lang: "load" },
  created() {
    this.load();
  },
  methods: {
    async load() {
      try {
        this.project = await api.project(this.id, this.lang);
        this.loadComments();
      } catch (e) {
        this.error = e.message;
      }
    },
    async loadComments() {
      try {
        this.comments = await api.comments(this.id, this.project.lang);
      } catch {
        this.comments = [];
      }
    },
    switchLang(code) {
      go(`#/watch/${encodeURIComponent(this.id)}/${code}`);
    },
  },
  template: `
    <main class="page watch">
      <div v-if="error" class="empty"><h2>This video is not available</h2><p>{{ error }}</p></div>
      <template v-else-if="project">
        <div class="watch-head">
          <h1>{{ project.title }}</h1>
          <div class="sub">{{ project.channel }} · dubbed into {{ project.langs.map(l => l.name).join(', ') }}</div>
        </div>
        <Player :media="project.media" :lines="project.lines" :speakers="project.speakers"
                :langs="project.langs" :lang="project.lang" :comments="comments.filter(c => !c.done)"
                @time="now = $event" @lang="switchLang" @comment="$refs.player.seek($event.t)" ref="player" />
        <div class="watch-comments">
          <CommentsCard :project-id="id" :lang="project.lang" :comments="comments" :now="now"
                        @changed="loadComments" @seek="$refs.player.seek($event)" />
        </div>
      </template>
      <div v-else class="hint" style="padding:40px;text-align:center"><span class="spinner"></span></div>
    </main>`,
};
