// A little steam locomotive, drawn facing right: the Dub Express's train.

export default {
  props: { size: { type: [Number, String], default: 40 } },
  template: `
    <svg class="loco" :width="size" :height="size * 0.62" viewBox="0 0 48 30" aria-hidden="true">
      <rect x="3" y="4" width="13" height="18" rx="2" fill="var(--loco-cab, #b73a26)" />
      <rect x="6" y="7" width="7" height="6" rx="1" fill="#f7c600" />
      <rect x="1" y="2" width="17" height="3" rx="1" fill="var(--loco-ink, #1d2433)" />
      <rect x="15" y="10" width="25" height="12" rx="5" fill="var(--loco-body, #123f6b)" />
      <rect x="32" y="4" width="4.5" height="7" rx="1" fill="var(--loco-ink, #1d2433)" />
      <rect x="23" y="7" width="4" height="4" rx="2" fill="#d9a21b" />
      <path d="M40 22l7 4H38z" fill="var(--loco-ink, #1d2433)" />
      <rect x="2" y="21" width="40" height="2.5" rx="1" fill="var(--loco-ink, #1d2433)" />
      <circle cx="9" cy="25" r="4" fill="var(--loco-ink, #1d2433)" /><circle cx="9" cy="25" r="1.4" fill="#d9a21b" />
      <circle cx="22" cy="25.5" r="3.5" fill="var(--loco-ink, #1d2433)" /><circle cx="22" cy="25.5" r="1.2" fill="#d9a21b" />
      <circle cx="32" cy="25.5" r="3.5" fill="var(--loco-ink, #1d2433)" /><circle cx="32" cy="25.5" r="1.2" fill="#d9a21b" />
      <path d="M9 25H32" stroke="#d9a21b" stroke-width="1.2" />
    </svg>`,
};
