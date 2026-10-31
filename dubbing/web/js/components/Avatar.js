// A speaker's picture: their face from the video when it is known for sure
// (server/avatars.py), otherwise a coloured circle with their number.

import { speakerColor } from "../util.js";

export default {
  props: { speaker: { type: Object, required: true }, size: { type: Number, default: 28 } },
  data: () => ({ broken: false }),
  computed: {
    style() {
      const s = this.size + "px";
      return { width: s, height: s, fontSize: Math.round(this.size * 0.42) + "px", "--c": speakerColor(this.speaker.id) };
    },
  },
  template: `
    <span class="avatar" :style="style" :title="'Speaker ' + (speaker.id + 1)">
      <img v-if="speaker.face && !broken" :src="speaker.face" alt="" @error="broken = true" />
      <span v-else>{{ speaker.id + 1 }}</span>
    </span>`,
};
