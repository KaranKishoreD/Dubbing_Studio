# Dubbing Studio

Paste a YouTube link and get the video back dubbed into an Indian language:
translated, spoken in voices that match the original speakers, timed to the
original, and lip-synced. Then ask for changes in plain words ("make line 5
sadder", "speaker 2 is an older man") and only the affected lines are redone.

**Languages (23):** all 22 scheduled languages of India, plus English. These
are Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati,
Punjabi, Odia, Assamese, Urdu, Nepali, Sanskrit, Maithili, Konkani, Dogri,
Kashmiri, Sindhi, Manipuri, Bodo and Santali. Every language has its own male
and female Bodhan voice. The first eleven, and English, can also use Sarvam
Bulbul voices. The last ten were added most recently and have had the least
real-world testing. Manipuri is written in Bengali script, Santali in Ol
Chiki, and Kashmiri and Sindhi in Perso-Arabic. Source videos can be in any
language Whisper understands.

---

## Highlights

### Features

- **23 languages in Indic mode** (all 22 scheduled Indian languages plus English) and **46 in International mode** (45 world languages plus English), from any source language Whisper understands.
- **A full dubbing pipeline from one link:** download, voice/background separation, word-level transcription, speaker and gender detection, transcript clean-up, translation, speech, timing, lip sync and mixing.
- **One consistent voice per speaker**, matched by pitch and gender, then **cloned to the original speaker's own timbre** (Seed-VC).
- **Natural, timed translation:** Bodhan indic-translate plus an LLM "dubbing director" that fixes gender agreement, spoken register and timing across the whole conversation.
- **Timing that never cuts words:** lines are rewritten shorter or longer by measuring the real audio, then fitted by pitch-preserving stretching (0.90x to as fast as needed, logged above 1.25x).
- **Lip sync** with LatentSync 1.6; the original music, effects and room tone are kept under the dub.
- **Edit in plain words:** "make line 5 sadder", "speaker 2 is an older man". Only the affected lines are redone.
- **Songs** are kept as sung, or optionally re-sung in the target language.
- **Fails soft:** every model has a fallback, and every stage caches its work.

### What is distinctive

1. **Smart lip sync: only the face that is really speaking.** Each face in each shot is scored with SyncNet against the original voice. Listeners, reaction shots and photos are left untouched ("when unsure, don't sync").
2. **Faces and voices together decide who speaks.** pyannote turns, a voiceprint second opinion, ArcFace face identities, two gender classifiers and face-based gender are combined, so even film scenes with many characters get the right speakers.
3. **Dub each person at their own pace:** speaker tempo (syllables per second) sets how much each line may say, and the real speaking rate of every voice is measured and remembered per language.
4. **A closed length-fitting loop:** budget before speech, measure after speech, rewrite and re-speak, then fit. Words are never cut.
5. **Natural-language editing that is validated:** one message becomes typed actions (about 15 kinds), each checked against the project, and only the necessary stages re-run.
6. **Versioned and undoable:** every change is a saved version that can be played or restored; nothing is lost.
7. **Hear before you apply:** audition one line in the new voice or wording before committing.
8. **Your own voice:** record a short clip and any speaker's lines are spoken in it.
9. **Names & words glossary** shared by the translators, the director and the transcript clean-up.
10. **Rate-limit aware:** a shared gate reads the speech service's own limits from its response headers and obeys retry-after, so long videos finish instead of failing.

### UI highlights

- **A railway-themed "Dub Express":** the ten pipeline steps are stops on a train route, each with a station board that says what the step does and which model runs it. Job progress is the same train moving stop to stop, with a time-left estimate learned from past runs.
- **Lotus, India map and globe language pickers:** click a language to start. International mode uses a real, draggable d3 globe with each language pinned where it is spoken.
- **Seamless compare:** Original, Dub and side-by-side Compare, with CC subtitles in the dub language, the original, or both.
- **Moment picker:** drag across a filmstrip or press `[` and `]` to select lines, then apply a change to exactly that stretch.
- **Pending changes bar:** edits queue up and wait for Apply, with Undo always available.
- **Search inside every dub:** press `/` to search titles and every line, original and dubbed, across the whole library.
- **Per-speaker voice cards** with audio previews, a Lines tab with Orig/Dub playback per line, and comments pinned to moments in the video.
- **Share links** that open a clean player with every language, a Download tab (MP4, M4A, subtitles), light and dark themes, and a built-in help sheet (`?`).
- **No build step:** the UI is plain Vue 3 modules served by the same FastAPI server.

---

## Contents

1. [What it does](#1-what-it-does)
2. [Requirements](#2-requirements)
3. [Accounts and API keys](#3-accounts-and-api-keys)
4. [Installation](#4-installation)
5. [Starting the studio](#5-starting-the-studio)
6. [Using the studio](#6-using-the-studio)
7. [Command-line tools](#7-command-line-tools)
8. [Settings](#8-settings)
9. [How it works](#9-how-it-works)
10. [Project layout](#10-project-layout)
11. [Troubleshooting](#11-troubleshooting)
12. [Costs and limits](#12-costs-and-limits)

---

## 1. What it does

| Step | What happens | Built with |
|---|---|---|
| Download | Fetches the video, or only the part you choose | yt-dlp |
| Separate | Splits the voice from music and background sound | Demucs |
| Transcribe | Writes down what is said, word by word with timings | faster-whisper large-v3, Bodhan indic-transcribe |
| Speakers | Finds who says each line, their gender and how they say it | pyannote, face analysis, voice classifiers |
| Clean up | Fixes recognition mistakes before translating | LLM via OpenRouter |
| Translate | Natural, spoken-style translation that fits the time available | Bodhan indic-translate + LLM "dubbing director" |
| Voice | Speaks every line, one consistent voice per speaker | Bodhan indic-speak or Sarvam Bulbul |
| Fit timing | Re-voices each line in the original speaker's own voice and fits it to the speaker's timing | Seed-VC, rubberband |
| Lip sync | Redraws the mouth on shots where the speaker's face is visible | LatentSync 1.6 |
| Export | Mixes the dub over the original music and effects | ffmpeg |

The original music, sound effects and room tone are kept; only the voices change.

---

## 2. Requirements

### Hardware

| | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA, **12 GB** memory (e.g. RTX 3060 12 GB, T4) | NVIDIA, **24 GB+** (RTX 3090 / 4090 / 5090, A10, L4, A100) |
| GPU generation | RTX 20-series (Turing) or newer | |
| RAM | 16 GB | 32 GB |
| Free disk | 60 GB | 100 GB |

- **Lip sync** needs about 10 GB of free GPU memory per worker. With 24 GB, 2
  workers run in parallel; with 12 GB, one.
- **Without a suitable GPU** (or with under 10 GB free) the studio still
  works, but lip sync is skipped. The dub is laid over the original video.
- **No NVIDIA GPU at all:** possible but very slow; see
  [CPU-only install](#cpu-only-install).

### Software

| | Version | Check with |
|---|---|---|
| Linux | Ubuntu 22.04+ or similar (Windows: use WSL2) | `lsb_release -a` |
| NVIDIA driver | **570 or newer** (needed for CUDA 12.8) | `nvidia-smi` |
| Python | **3.11** exactly | `python3.11 --version` |
| ffmpeg | any recent | `ffmpeg -version` |
| git | any | `git --version` |
| rubberband | optional, for better time-stretching | `rubberband --version` |

You do **not** need to install CUDA itself; it comes with the Python packages.

Install the system packages on Ubuntu:

```bash
sudo apt update
sudo apt install -y git ffmpeg rubberband-cli python3.11 python3.11-venv
```

If `python3.11` is not available on your Ubuntu version:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update && sudo apt install -y python3.11 python3.11-venv
```

---

## 3. Accounts and API keys

Create these before installing. All of them have free sign-up.

| Key | Required? | Used for | Where to get it |
|---|---|---|---|
| `BODHAN_TRANSLATE_KEY` | **Yes** | Translation | [console.bodhan.ai](https://console.bodhan.ai) → create a key for *indic-translate* |
| `BODHAN_SPEAK_KEY` | **Yes** | Text to speech | Same console → key for *indic-speak* |
| `OPENROUTER_API_KEY` | **Yes** | Transcript clean-up, dubbing director, length fitting, Ask tab | [openrouter.ai/keys](https://openrouter.ai/keys) (add a few dollars of credit) |
| `BODHAN_TRANSCRIBE_KEY` | Recommended | Better transcripts for Indic-language videos | Same console → key for *indic-transcribe* |
| `HF_TOKEN` | Recommended | Accurate speaker detection (pyannote) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → *Read* token |
| `SARVAM_API_KEY` | Optional | Extra voices (Sarvam Bulbul, 11 languages) | [dashboard.sarvam.ai](https://dashboard.sarvam.ai) |

**One extra step for `HF_TOKEN`:** while logged in to Hugging Face, open
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
and accept the terms. Without this, speaker detection falls back to a simpler
method.

---

## 4. Installation

Expected time: **1–2 hours**, almost all of it downloading (about 25 GB of
packages and models). On a slow connection it can take several hours. Run it
in `tmux` or `screen` so it survives a dropped SSH session.

### Step 1: Get the code

```bash
git clone https://github.com/KaranKishoreD/Dubbing_Studio.git
cd Dubbing_Studio
```

### Step 2: Run the setup script

```bash
bash scripts/setup_gpu.sh
```

What it does, in order:

1. Checks that ffmpeg is installed (and rubberband, optionally).
2. Creates the Python environment `.venv/` and installs the packages from
   `dubbing/requirements.txt` and `dubbing/requirements-gpu.txt`.
3. Checks that your GPU really works with this PyTorch build.
4. Creates `.env` from `dubbing/.env.example` if it does not exist yet.
5. Asks: **"Clone LatentSync for lip sync?"** Answer **y** (about 5 GB). Say
   N only if you do not want lip sync.
6. Asks: **"Clone Seed-VC for speaker voice cloning?"** Answer **y** (about
   1 GB). This makes the dub sound like the original speakers.
7. Sets up the vocal separator used for songs.
8. Downloads the Whisper and Demucs models in advance.

When it finishes it prints `Setup complete.`

### Step 3: Add your keys

Open `.env` in the project root (the setup script created it) and paste the
keys from [section 3](#3-accounts-and-api-keys):

```bash
nano .env
```

```ini
BODHAN_TRANSLATE_KEY=...
BODHAN_SPEAK_KEY=...
OPENROUTER_API_KEY=...
BODHAN_TRANSCRIBE_KEY=...
HF_TOKEN=...
```

Leave every other line as it is.

### Step 4: Check the install

One command checks the system tools, Python environment, GPU, optional
models, API keys (names only) and the tests, and changes nothing:

```bash
bash scripts/check_setup.sh
```

Or by hand:

```bash
source .venv/bin/activate
cd dubbing
python -m pytest -q
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

The tests should say `passed`, and the second command should print your GPU's
name.

### CPU-only install

For a machine without an NVIDIA GPU (much slower: expect over an hour per
minute of video, and no lip sync):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r dubbing/requirements.txt
cp dubbing/.env.example .env      # then add your keys
```

---

## 5. Starting the studio

Every time:

```bash
cd Dubbing_Studio
source .venv/bin/activate
cd dubbing
python -m server
```

Then open **http://localhost:8000** in a browser on the same machine.

| Option | Meaning |
|---|---|
| `python -m server --port 9000` | Use another port. |
| `python -m server --host 0.0.0.0` | Reachable from other computers on the network. |

To stop the studio press `Ctrl+C`. Dubs that were running stop too; finished
work is kept and reused next time.

### GPU machine is a remote server

Forward the port over SSH from your own computer, then open
`http://localhost:8000` there:

```bash
ssh -L 8000:localhost:8000 user@your-server
```

### Sharing a public link

1. Set a password in `.env`, since anyone with the link could otherwise use
   your API credits:
   ```ini
   STUDIO_PASSWORD=choose-something
   ```
   The browser then asks for it (user name `studio`).
2. Start the studio, and in a second terminal start a free Cloudflare tunnel
   ([install `cloudflared`](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)):
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
3. Share the `https://….trycloudflare.com` address it prints. It works as
   long as both commands keep running.

---

## 6. Using the studio

### Library (home page)

At the top, the language stage, which you can flip between two views:

- **🪷 Lotus.** A 3D lotus opens and closes on its lily pad, with every
  language orbiting it in its own script. The orbit turns on its own. Drag it
  to spin it, or use ← → when it has focus. The language passing in front
  leads the headline ("Make any video speak தமிழ்"). The pond tilts towards
  the mouse.
- **🗺️ Map.** India drawn in kolam dots, with a pin where each language is
  spoken. Pointing at a pin lights up its region, dot by dot outwards from
  the pin.

Clicking a language in either view opens New dub with that language already
picked. Paste a link into the box beside it to start. On the home page this
replaces the top-bar **+ New dub** button. *How a dub is made* shows the
ten pipeline steps as the stops of the "Dub Express", which runs north to
south along the Himsagar Express route, from Jammu Tawi to Kanyakumari. Each
stop has a yellow Indian Railways board that says what the step does (Hindi
above, English below, for example "कौन बोला · WHO SPEAKS"). The real station
and its code are only the small print. Under each board are the step in plain
words and the model that runs it. The boards stand along one unbroken track
that snakes from the top right, round a curve, to the last stop. You drive
the train: it waits at the start, and a click anywhere on the panel (or **Next
stop**, or the → key) takes it one stop on. **Back** and **Ride to the end** are
there too. Each board comes into
focus when the train arrives. The stops still ahead are blurred, with a brief
glitch now and then, so what happens there stays a surprise.

Below, every dub made so far is a card in a grid, newest first and grouped
by day under ornamental dividers ("Today · आज", "This week · इस हफ़्ते", …).
Every card is the same size, with a short, uncropped-looking picture in a thin gold frame with diamond
corners, a block-print scalloped edge, and the details on a jewel tint
(blush, butter, mint or sky) with a matching coloured base. Click a card to
open the dub, or one of its round language badges (हि, த, …) to open that
language directly. Dubs being made right now come first, under "Being made",
with the train inside a progress ring and the current step.

**Finding a dub.** Press <kbd>/</kbd> and type. The search box matches
titles, channels and language names (in English or in their own script). It
also searches **every line said in every dub**, both the original and the
dubbed text. Each matching line is listed under its dub with its time.
Clicking a line opens that dub at that line. The language beads under the
box filter the thread to one language. Typing a link offers to dub it.
Esc clears the search.

The look is *rani aur mor* (rani pink and peacock), in saturated jewel
tones. The page is a deep peacock teal patterned with gold bandhani dots, with
large abstract shapes that scroll with the page (`web/img/backdrop.svg`, a
tile repeated down the page): a marigold sun, a rani paisley,
mandala rings, a jharokha arch, a royal-blue river band and scattered gold
diamonds. The cards in front are tinted, not plain white. The hero sits on
muted, softer tones (warm sand behind the words, dusky rose behind the
lotus), with a slowly turning mandala in line art and a block-print border.
The Dub Express panel is ivory. Your dubs alternate mint, blush, butter and
sky, each with an edge in its jewel colour. In the studio, the moment
picker is a deep peacock "editing desk" with cream text. The script, voices,
Ask, Versions and Download panels are soft khadi-green paper under a rani tab
bar, so they are easy on the eyes. Actions and highlights are rani pink, with marigold and emerald beside
it. The header is a deep rani gradient with a toran (string of flags) of
marigold, emerald, white and royal blue. The ☾/☀ button in the header switches to a
dark mode with the same jewels, deeper and quieter. The choice is remembered;
until you choose, the computer's own setting is used. Simple line icons are
used instead of emoji. The page builds itself in on arrival: the hero text
rises in, the language name wipes in in its reading direction (right to left
for Urdu, Kashmiri and Sindhi), and sections fade up as you scroll. Job progress is the same railway line, with a train
moving from stop to stop and each stop's station code in small print. An edit runs as an express that stops only
at the stations it needs.

### Making a new dub

1. Paste a link on the home page (or click a language on the lotus or map).
   On other pages, press **+ New dub** (top right).
2. Paste a **video link** from YouTube or any site yt-dlp supports (Vimeo,
   Instagram, X, Facebook, Dailymotion, TED…), or **drop a file** (MP4,
   MOV, MKV; MP3/WAV audio is dubbed without lip sync). The title and
   picture appear when the link works.
3. Pick the language under **Dub into**. Tick **Several languages** to make
   more than one: they run one after another, and download, transcript and
   speaker detection are done once and shared.
4. Optional: fill in **Only part of the video** (e.g. `1:30` to `3:00`).
   Only that part is downloaded. Good for a quick first try.
5. Optional: untick **Lip sync** for fast mode (see below): much quicker,
   and lip sync can be added later.
6. Optional: **More options**:

   | Option | What it does |
   |---|---|
   | **Speakers** | Number of people speaking. *Detect automatically* works for most videos; setting it is the most reliable. |
   | **Content** | Keeps the right technical terms (lecture, medicine, law, …). |
   | **English words in the dub** | *None*: translate everything. *Natural*: keep the English words people really say ("phone", "meeting"). *Heavy mix*: Hinglish/Tanglish. |
   | **Translation** | *Indic + LLM polish* (default): Bodhan translates, an LLM fixes gender, tone and timing. *LLM only*: an LLM translates the whole conversation. |
   | **Songs** | *Keep original singing* (default), or *Re-sing* in the new language (experimental). |
   | **Voice engine** | Bodhan (all languages) or Sarvam (needs `SARVAM_API_KEY`). |

7. Press **Start dubbing**. A progress page shows each stage. You can leave
   it: the dub keeps running, a pill in the top bar shows its progress, and it
   opens when done.

Typical time on a 24 GB GPU: **3–5 minutes per minute of video** without lip
sync, **10–15 minutes** with lip sync. The first run is slower because models
download. Jobs run one at a time; extra jobs wait in a queue.

### Indic and International mode

The home page has an **Indic / International** switch above the headline, and
the choice is remembered. In Indic mode the language picker is the lotus or
India map. In International mode it's a real **globe**: Natural Earth land, coastlines and
country borders, drawn with d3 on a canvas (1:110m while it spins, 1:50m when
zoomed). It turns to each language in turn. Each language is pinned everywhere
it is widely spoken, so English appears in the US, UK, Australia and Nigeria,
Spanish across Latin America, Portuguese in Brazil, and French in Canada and
West Africa. Drag to turn it, scroll or use +/− to zoom (**Europe** jumps
straight to the crowded part), point at a pin to stop it, and click a pin to
start a dub, which opens New dub already in International mode.

The **New dub** box opens with a switch:

| | Indic (default) | International |
|---|---|---|
| Dub into | the 22 scheduled Indian languages and English | English and **45 world languages** (Spanish, French, German, Russian, Arabic, Japanese, Chinese, Korean, Turkish, …) |
| Transcribe | Whisper large-v3, plus Bodhan indic-transcribe for Indian-language sources | Whisper large-v3 |
| Translate | Bodhan indic-translate, then the LLM dubbing director | the LLM (Claude Sonnet via OpenRouter) |
| Voices | Bodhan indic-speak, or Sarvam Bulbul | **Kokoro-82M** (Apache 2.0) where it has the language and gender, otherwise **Piper** (rhasspy, 53 languages), both open and local |

Speakers, voice cloning (Seed-VC), lip sync and the mix are the same in both.
Piper voices download on first use of a language (about 60 MB each, up to 4
per language) into `dubbing/voices/piper/`. Each voice's gender is measured
once with the same voice classifier used for speakers. A language with voices
of only one gender reads everyone with them, and Seed-VC then clones each
actor's own voice. Speech in any language Whisper understands can be dubbed in either
mode.

### Lip sync: only the face that is really speaking

Lip sync never decides by who is on screen. Each line is cut into shots at
camera cuts. In every shot, each face (turned at most ~40°) is cropped and
scored with SyncNet (LatentSync's `syncnet_v2`) against the *original* voice
of that moment (`pipeline/video/smart_sync.py`). A face whose lips line up
with the voice (offset within ±4 frames, confidence ≥ 1.5) is synced. If none
does (a listener, a reaction shot, a speaker off-screen), the shot keeps its
original frames. Shots under 0.9 s are not scored. If scoring fails, the
older screen-time planner is used.

### Working on a dub

Opening a video shows the player and timeline on the left and five tabs on
the right: **Lines**, **Voices**, **Ask**, **Versions** and **Download**.

**Where to start editing.** At the top of the right side, a three-step strip
shows where you are (① Pick what to change → ② Choose the change → ③ Apply).
One line under it says that changes wait in a list until you press
**Apply**. Below that, *What would you like to change?* offers six plain
choices. Each takes you to the place where that change is made and briefly
highlights what to click there:

| Choice | Takes you to |
|---|---|
| A part of the video | the moment picker (drag across the film strip) |
| The words of a line | Lines, with ✎ Dub and ✎ Transcript lit up on every line |
| Someone's voice | Voices |
| Lip sync | adds "lip sync the whole video again" to your changes |
| A name said wrong | Ask, with the Names & words list open |
| Something else | Ask, with the cursor in the request box |
| Go back | Versions |

**Hide** folds the choices away, and *What can I change?* brings them back.

**Lip sync as an edit.** Lip sync can be redone on its own, with the audio
left exactly as it is. You can pick lines and use the **Lip sync** tile,
choose **Lip sync** in the guide for the whole video, or ask in your own
words ("there is no lip sync in the first minute"). When lip sync is the only
change, just the lip sync and final mix steps run, so nothing is translated
or voiced again. Mixed with other changes, those lines are lip-synced in the
same run. Asking for lip sync runs it even if the dub was made in fast mode.

**Player**
- **Original / Hindi / Tamil …** switches between the original and every
  language the video is dubbed in, at the same moment. Keyboard: `O`
  toggles the original.
- **⇆ Compare** shows the original and the dub side by side, split by a
  handle you can drag, playing in sync with the dub's sound. It is the
  quickest way to see the lip sync. Keyboard: `C`.
- The current dubbed line is shown over the video.
- The **moment picker** (the timeline) shows a filmstrip of the video with
  one row per speaker under it, with their face when it is clear from the
  video, otherwise their number. Hover to see that frame and line; **click**
  to jump there.

**Picking a moment.** Drag across the filmstrip, or press `[` and `]` while
it plays (or **⇤ Mark start here** / **Mark end here ⇥**). The moment gets
two handles to drag; edges snap to where lines start and end. **🔁 Play this
moment** (`L`) loops it. You can also tick lines in the script (shift-click
selects a run). A **What should change here?** card appears with big tiles
for just those lines, with no typing of line numbers or times:
- **Voice**: another voice (press ▶ to hear each). If they are all of one
  speaker's lines, that speaker gets the voice; otherwise the lines become a
  speaker of their own. **The speaker's own steady voice** re-clones just
  these lines from the speaker's clean voice, for lines that sound like
  someone else or the wrong gender.
- **Who's speaking**: an existing speaker, or a new person (man or woman).
  This fixes speakers the detection merged.
- **Feeling**: happy, sad, angry, surprised, warm, storytelling, news…
- **Language**: dub just these lines in another language (e.g. Bengali);
  the rest stays in the dub's language, and each speaker keeps their voice.
- **Keep original** / **Dub again**.
- **In your words**: anything else, e.g. "more casual"; it is planned for
  these lines only (see the Ask tab).

**Changes wait until you apply them.** Every fix (from the selection bar,
the Voices tab, a line edit or the Ask tab) is added to a list at the top of the
side panel, and the lines it will redo are marked "● changing". Remove any
item with ✕, then **Apply all & re-dub** runs them together as one job that
redoes only the affected lines. A later choice for the same thing (a
speaker's voice, a line's wording) replaces the earlier one, lines can be
given to a new person already in the list, and the list survives a page
reload. Above the list, **🎯 Will redo …** says exactly which lines applying
would redo and about how long it takes. Only **Find speakers again** runs at
once, since it redoes the whole dub.

**Lines tab.** Every line with its speaker, voice, original and translation.
It follows playback, and clicking a line jumps there. Hover a line for:
**▶ Orig** and **▶ Dub** to hear just that line, and **✎ Dub** to type new
wording, and **✎ Transcript** to correct what was said (a mis-heard word,
words nobody said, unwanted words); the line is then translated again. Both
are added to your changes; only that line is redone, lip sync included.
Lines in another language show a 🌐 tag. Marks: **⚠ check**, transcript may
be wrong; **＋ recovered**, speech first missed; **♪ sung**.

**Voices tab.** One card per speaker. Pick another voice, press **▶** to hear
it, then add it to your changes. Fixes one click away: **Steady voice** (their voice
changes from line to line: every line is re-cloned from their own clean
voice), **Actually a man/woman** (wrong gender detected) and **Same person
as…** (merge two speakers). If speakers are mixed up throughout, **Find
speakers again** (optionally with the number of people) detects them afresh
and dubs the video again; download, separation and transcription are reused,
and the current dub stays in History.

**Ask tab.** Describe any number of changes in one message, e.g.
*"make the intro more casual, speaker 2 is an older man with a deeper voice,
lines 4 to 6 are a different woman, don't dub the song"*. **Plan changes**
shows how each request was understood, and anything it could not do;
**Add to changes** puts the plan in your list. Vague
requests ("X's voice became female", "X has several voices") get the most
likely fix, with the guess stated, since you review the plan before it runs. With lines selected, the
request applies to those lines only.

**↶ Undo** (top right) reverses the last change. The lines the last change
redid glow green on the timeline.

**Versions tab.** Every change made to this dub, newest first, in plain words,
with the lines it touched (click one to jump there). **▶ Play this version**
plays it from its first changed line; **↶ Go back to this** makes it the
current dub again. Going back is itself saved as a version, so nothing is
ever lost.

**Names & words** (top of the Ask tab). Names, brands and terms the dub
must get right: leave "say it as" empty to keep a word as it is (Muthuvel,
convex set), or fill it in (Rajini → रजनी). Every translation and the
transcript clean-up follow the list; after saving, **Re-translate just
those** redoes only the lines that use the words.

**Comments.** Anyone with the studio or a watch link can leave a comment on
the current moment (or a marked one) with their name. Comments show as 💬 on
the timeline and in the Ask tab, where **Send to Ask** puts the
comment in the request box, with the lines at that moment selected, to edit
and plan; **✓ Done** files it away.

**Subtitles.** The **CC** button on the video cycles through the dub
language, the original, both, and off. The Download tab also offers the
dubbed video with subtitles burned in.

**Your own voice.** In the Voices tab, **🎤 Use my voice** on a speaker:
record 15–30 s reading the paragraph shown (or upload a recording), say
whether it is a man's or a woman's voice, and add it to your changes. That
speaker's lines are then cloned from your recording. Useful when creators
dub their own videos into languages they don't speak (they still sound like
themselves), for a narrator, or to replace a noisy original recording.

**Hearing a change first.** In the "What should change here?" card, the ▶
next to every voice, feeling and language plays the selected line made
exactly that way (one speech call; replaying is free). A voice you pick for
a speaker is kept as it is; everyone else is cloned to sound like the
original person.

**Help.** Press `?` (or the **?** button) for how to do the common things and
every keyboard shortcut. The first time a dub opens, a short welcome guide
shows the three steps.

**Download tab.** The dubbed video (MP4), the soundtrack alone (M4A), and
subtitles in both languages (SRT).

**↗ Share** copies a link to a clean player for this video (`#/watch/…`):
every dubbed language, the original and Compare, without the editing tools.

**While a dub runs**, its page shows the time left and why a step may be
slow (for example, a busy GPU), then the lines as they are made: transcript,
translation, and a ▶ on each line as soon as its voice exists.

**Fast mode (lip sync off).** Lip sync is the slowest step (LatentSync 1.6,
GPU). Untick **Lip sync** in the New dub form, or next to **Apply all** in
the changes list, to skip it: everything else is made as usual and the
mouths keep their original movement. The choice is remembered per dub.
Lines made this way are counted in a banner on the dub's page; **Add lip
sync** later renders just those lines and reuses every finished shot.
With lip sync on, an edit renders only the lines it changed.

**■ Stop** (on the progress bar) cancels a waiting job at once, and stops a
running one as soon as its current step finishes. No step is cut off
halfway, so no half-made file is reused later, and the final step always
completes because it writes the video. Nothing in the dub changes; finished
steps are kept, so starting again is quicker. A stopped new dub offers
**Change settings & start again** with the form filled in as before; a
stopped edit puts its changes back in the waiting list to adjust.

Dubs made before the web studio existed may show *"can be watched but not
edited"*. Press **Dub again**: finished steps are reused, and it becomes
editable.

All results are saved under `dubbing/cache/<video id>/`, including version
history (`versions/`); there is no database.

### Logs and API

The full log of every run is in `logs/ui_debug.log`. The API is documented at
`http://localhost:8000/api/docs`.

---

## 7. Command-line tools

Run these from the `dubbing/` folder with the environment active.

| Command | What it does |
|---|---|
| `python -m tools.list_voices` | Checks which Bodhan voice names your key accepts. |
| `python -m tools.build_voice_profiles hi "नमस्ते, यह एक परीक्षण है।"` | Measures the pitch of every voice, so casting can match voices to speakers. Run once per new voice catalogue. Results go to `cache/_voice_profiles.json`. |
| `python -m server` | Starts the web studio (section 5). |
| `python -m pytest -q` | Runs the tests: every module imports and every internal import resolves. |
| `ruff check . && ruff format --check .` | Lint and formatting checks (settings in `dubbing/pyproject.toml`). |

---

## 8. Settings

Everything works with the defaults. To change something, uncomment its line in
`.env`; `dubbing/.env.example` lists the useful ones with explanations. The
most common:

| Setting | Default | Meaning |
|---|---|---|
| `TRANSLATE_MODEL` | `anthropic/claude-sonnet-5.5` | Any OpenRouter model id. |
| `VOICE_CLONE` | `1` | Re-voice lines in the original speaker's voice. |
| `LIPSYNC_BACKEND` | automatic | `latentsync`, or `none` to skip lip sync. |
| `LATENTSYNC_MAX_WORKERS` | `4` | Upper limit on parallel lip-sync workers. |
| `MAX_STRETCH` | `1.10` | Speed-up applied without rewriting a line. |
| `MAX_VIDEO_HEIGHT` | `720` | Download resolution. |
| `YTDLP_COOKIES_BROWSER` | (none) | Use your browser's YouTube login if YouTube asks you to sign in. |


---

## 9. How it works

The pipeline is a [LangGraph](https://github.com/langchain-ai/langgraph) graph
of ten stages (`dubbing/pipeline/graph.py`). Each stage is one file in
`dubbing/pipeline/steps/`, whose opening comment lists its steps.

```
download → separate → transcribe → diarize → refine → translate
         → synthesize → align → lipsync → assemble
```

There are four graphs:

- **Full:** all ten stages, for a new video.
- **Edit:** from *translate* on, after a change request.
- **Re-voice:** from *synthesize* on, after a cast change.
- **Lip sync:** just *lipsync → assemble*, for an edit that only redoes lip sync.

Every stage saves its work, so a re-run redoes only lines whose input changed.

### Key ideas

- **Faces split speakers only when the voice agrees.** After voice
  diarization, a second talking face on one speaker's lines can mark a
  character the voices missed. In films, though, the camera often shows the
  listener, or the same actor from another angle. A face-based split is
  therefore made only when the two groups of lines also sound different:
  their ECAPA voiceprint similarity must be below 0.35, tunable with
  `FACE_SPLIT_SAME_VOICE`. On a *3 Idiots* scene, the same actor across
  shots scored 0.51–0.64 and different people 0.00–0.22. The check took that
  scene from 5 speakers down to the correct 3.
- **Reactions are kept.** Laughter, claps, cheers and side comments often
  overlap a line, and they live in the voice stem, so swapping the dialogue
  for the background bed used to remove them. The final mix now splits the
  voice stem once more, with a Mel-RoFormer karaoke model, into the main
  speaker and everyone else. "Everyone else" goes back under the dub at its
  original timing and level. Where that layer is more than 20 dB quieter
  than the main speaker it is only separation leak, so it is muted there to
  keep the original language out (`REACTIONS_LEAK_DB`). The layer is
  cached per video, and `REACTIONS=0` turns it off.
- **Speed is its own edit.** "Faster", "slower", "rattling it off" become
  `set_pace`. It picks the delivery style measured to be fastest (or
  slowest) for that line's voice and language, rather than a style chosen by
  topic. The studio offers it as Faster / Normal speed / Slower under
  Feeling.
- **Speakers first, sentences later.** Speaker changes are found word by
  word before lines are joined into sentences, so one line never mixes two
  people. Faces on screen split characters whose voices sound alike.
- **Translate, then direct.** Bodhan translates line by line. An LLM
  "director" then reads the whole conversation and fixes what one line at a
  time cannot see: the speaker's gender in verb endings, spoken rather than
  written style, consistent forms of address.
- **Fit the time, never cut words.** Each line gets a time budget from how
  fast that person speaks. Lines too long are rewritten shorter; small
  differences are absorbed by gentle speed changes; nothing is cut off.
- **Sound like the speaker.** Each line is spoken by a TTS voice matched to
  the speaker's gender and pitch, then converted to the speaker's own voice
  with Seed-VC. A conversion may change the voice's timbre, never its gender:
  if music or another voice under a line pulls the clone into the other
  gender's range, it is redone from the speaker's clean reference, or the TTS
  voice is kept.
- **Lip sync only where it shows.** A shot is lip-synced only if the
  speaker's face is visible and their mouth is moving. Synced shots are laid
  over the original video, so everything else stays untouched.
- **Keep the soundtrack.** The voice is removed with Demucs and the music,
  effects and room tone are kept under the dub, with dialogue lifted where
  music is loud.

---

## 10. Project layout

The project has three parts: the **dubbing engine** (`pipeline/`), which turns
a video into a dubbed video; the **web server** (`server/`), which runs the
engine and keeps every project; and the **web page** (`web/`), which you use
in the browser. Each folder's `__init__.py` says in a few lines what the
folder is for.

```
Dubbing_Studio/
├── README.md                 this file
├── .env                      your API keys (not in git; copy dubbing/.env.example)
├── scripts/setup_gpu.sh      one-time setup: environment, models, helper tools
└── dubbing/
    ├── pipeline/             THE DUBBING ENGINE
    │   ├── graph.py            the order of the steps (full dub, edit, re-voice, lip sync)
    │   ├── state.py            the data handed from step to step (one entry per line)
    │   ├── cache.py            where each project's files are kept
    │   ├── bodhan_gate.py      keeps Bodhan API calls within its rate limits
    │   ├── steps/              the ten steps, one file each, in running order
    │   │   ├── download.py       1  get the video (any link, or an uploaded file)
    │   │   ├── separate.py       2  split the voices from music and background
    │   │   ├── transcribe.py     3  write down what is said (Whisper)
    │   │   ├── diarize.py        4  work out who says each line
    │   │   ├── refine.py         5  tidy the transcript (LLM)
    │   │   ├── translate.py      6  translate, then polish the whole dialogue
    │   │   ├── synthesize.py     7  speak each line in the new language
    │   │   ├── align.py          8  make each line sound like the actor and fit its time
    │   │   ├── lipsync.py        9  move the speaker's lips to the new words
    │   │   └── assemble.py      10  mix the soundtrack and put the video together
    │   ├── speech/             understanding the original speech
    │   │   ├── speaker_turns.py  who spoke when (pyannote + voiceprints)
    │   │   ├── gender.py         man or woman, to pick a matching voice
    │   │   ├── pace.py           how fast each person talks
    │   │   ├── prosody.py        how each line was said (calm, energetic…)
    │   │   └── bodhan_asr.py     a second transcription for Indian-language videos
    │   ├── text/               the words
    │   │   ├── languages.py      language names, scripts, English-word rules
    │   │   ├── line_by_line.py   Bodhan translation, one line at a time
    │   │   ├── dialogue.py       LLM translation of the whole conversation
    │   │   ├── director.py       the "dubbing director": one LLM pass that polishes it all
    │   │   ├── length_fit.py     makes each line short enough to say in its time
    │   │   ├── transliterate.py  English words into the target script, for the voices
    │   │   ├── glossary.py       "Names & words" the dub must keep right
    │   │   ├── llm.py            the connection to the LLM (OpenRouter)
    │   │   └── json_utils.py     reading the LLM's answers
    │   ├── voice/              the new voices
    │   │   ├── catalogue.py      every voice that exists (Bodhan, Sarvam, Kokoro)
    │   │   ├── casting.py        which voice each speaker gets
    │   │   ├── styles.py         the speaking style of each line
    │   │   ├── emotion.py        which lines carry real feeling
    │   │   ├── tts.py            text → speech, one line at a time
    │   │   ├── piper_voices.py   Piper voices for 53 languages (International mode)
    │   │   ├── tts_rate.py       how fast each voice speaks
    │   │   ├── refit.py          re-fits lines that came out too long or short
    │   │   ├── voice_clone.py    makes each line sound like the actor (Seed-VC)
    │   │   └── seedvc_worker.py  runs Seed-VC in its own environment
    │   ├── video/              faces and lip sync
    │   │   ├── faces.py          finds faces and who is talking on screen
    │   │   ├── smart_sync.py     picks the shots where a face really says the line
    │   │   ├── syncnet_worker.py scores how well lips match a voice (SyncNet)
    │   │   ├── shot_planner.py   the older way of picking shots (fallback)
    │   │   ├── latentsync_runner.py  runs LatentSync, the lip-sync model
    │   │   ├── latentsync_worker.py  LatentSync's side, in its own environment
    │   │   ├── legacy_lipsync.py older lip-sync options (Wav2Lip, fal.ai)
    │   │   ├── frame_interp.py   matches frame rates of lip-synced shots
    │   │   └── compose.py        puts lip-synced shots back into the video
    │   ├── audio/mixing.py     levels and layering for the final mix
    │   ├── music/              songs
    │   │   ├── song_detect.py    finds sung lines (kept in the original by default)
    │   │   └── song_mode.py      re-sings them in the new language (experimental)
    │   └── edits/edit_planner.py   a plain-words request → a checked list of edits
    │
    ├── server/               THE WEB SERVER  (start with: python -m server)
    │   ├── __main__.py         starts it on port 8000
    │   ├── startup.py          loads your keys from .env and starts the log file
    │   ├── app.py              every web address the page calls (the API)
    │   ├── options.py          languages, steps, voices and styles shown in forms
    │   ├── runner.py           builds a dub's settings and runs the engine step by step
    │   ├── jobs.py             the queue: one dub at a time, with live progress
    │   ├── eta.py              time estimates, learned from past runs
    │   ├── projects.py         the library: every dub, read from its folder
    │   ├── versions.py         version history, undo and go back
    │   ├── revise.py           the Ask tab: request → plan, and "what changed"
    │   ├── audition.py         ▶ Try: hear one line with a change before applying
    │   ├── exports.py          the Download tab's files
    │   ├── uploads.py          uploaded files and links to other sites
    │   ├── comments.py         review comments on a dub
    │   └── avatars.py          a face picture for each speaker
    │
    ├── web/                  THE WEB PAGE  (Vue 3, no build step)
    │   ├── index.html          the page itself
    │   ├── css/studio.css      all the styling, both themes and both modes
    │   ├── img/                the background patterns (Indic and International)
    │   ├── vendor/             Vue, d3 and the world map data (kept locally)
    │   └── js/
    │       ├── main.js           starts the page: top bar, dark mode, dialogs
    │       ├── store.js          shared state and page addresses (#/, #/p/…)
    │       ├── api.js            calls to the server
    │       ├── util.js           small helpers (times, languages, stations)
    │       ├── views/            the four pages
    │       │   ├── Library.js      home: the switch, lotus/map/globe, your dubs
    │       │   ├── Project.js      one dub: player, timeline and the five tabs
    │       │   ├── JobPage.js      a dub being made
    │       │   └── Watch.js        a clean player to share
    │       └── components/       pieces of those pages
    │           ├── NewDubDialog.js   the New dub form
    │           ├── BhashaStage.js    the lotus and India map (Indic mode)
    │           ├── GlobeStage.js     the world globe (International mode)
    │           ├── DubExpress.js     "How a dub is made", the train journey
    │           ├── JobProgress.js    the progress line while a dub runs
    │           ├── Player.js         video player, timeline, moment picker
    │           ├── EditGuide.js      "What would you like to change?"
    │           ├── SelectionBar.js   the change tiles for selected lines
    │           ├── PendingBar.js     changes waiting to be applied
    │           ├── LinesPanel.js     Lines tab
    │           ├── VoicesPanel.js    Voices tab
    │           ├── AskPanel.js       Ask tab
    │           ├── VersionsPanel.js  Versions tab
    │           ├── DownloadPanel.js  Download tab
    │           ├── GlossaryCard.js   Names & words
    │           ├── CommentsCard.js   review comments
    │           ├── VoiceRecorder.js  "Use my voice"
    │           ├── HelpSheet.js      the help sheet (press ?)
    │           ├── Avatar.js         a speaker's picture
    │           ├── Icon.js           line icons
    │           └── Loco.js           the little train
    │
    ├── tools/                one-off scripts for the voice catalogue (section 7)
    ├── tests/test_imports.py   checks every file loads and every import is right
    ├── requirements.txt        Python packages
    ├── requirements-gpu.txt    the GPU-only packages
    ├── pyproject.toml          lint settings
    └── .env.example            which keys go in .env
```

Created by setup and not in git: `.venv/`, `dubbing/LatentSync/`,
`dubbing/SeedVC/`, `dubbing/Separator/` (the three big models, each in its own
environment), `dubbing/voices/` (downloaded Piper voices), and `dubbing/cache/`
(every project's files).

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `BODHAN_TRANSLATE_KEY not set` on start | `.env` must be in the project root (next to `README.md`), not in `dubbing/`. |
| `CUDA error: no kernel image is available` | Your driver or PyTorch is too old for the GPU. Update the NVIDIA driver to 570+, then `pip install --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu128 torch torchaudio torchvision`. |
| `Sign in to confirm you're not a bot` | Set `YTDLP_COOKIES_BROWSER=firefox` (or chrome) in `.env`, or export cookies to a file and set `YTDLP_COOKIES_FILE`. |
| Log says *lip sync skipped (not enough GPU memory)* | Another program is using the GPU. Check with `nvidia-smi`. The dub is still made, without lip sync. |
| Everyone is one speaker | Set **Speakers** to the right number. Check `HF_TOKEN` and that the pyannote terms are accepted. |
| A voice has the wrong gender | Change it in the **Voices** tab, or in **Ask**: *"speaker 2 is a woman"*. |
| `401` from Hugging Face | Accept the model terms (section 3) with the same account as the token. |
| First run seems stuck | Models are downloading (several GB). Press **Show details** on the progress bar, or watch `logs/ui_debug.log`. |
| The page asks for a password | `STUDIO_PASSWORD` is set in `.env`; user name `studio`. |
| Video will not play over a shared link | Use the Cloudflare tunnel (section 5), not an SSH tunnel through a slow jump host. |
| `ffmpeg: not found` | `sudo apt install ffmpeg`, then restart the studio. |

The full log of every run is in `logs/ui_debug.log`.

---

## 12. Costs and limits

**Costs.** A few-minute video typically costs well under $1 in OpenRouter
credits (Claude Sonnet). Bodhan and Sarvam charge per character from their
own credits. Re-runs reuse saved results and cost almost nothing. Re-singing
songs costs about $0.04 per sung phrase.

**Known limits.**

- Re-sung songs get a new tune; the original melody is not kept yet. The
  default keeps the original singing.
- Lip sync works best on faces turned less than about 40° from the camera.
  Other shots keep their original frames.
- Very noisy audio, or many people talking over each other, makes speaker
  detection less reliable. Setting the number of speakers helps.
- Long videos take proportionally longer; try a clip range first.
