# Dubbing Studio

Paste a YouTube link and get the video back dubbed into an Indian language:
translated, spoken in voices that match the original speakers, timed to the
original, and lip-synced. Then ask for changes in plain words ("make line 5
sadder", "speaker 2 is an older man") and only the affected lines are redone.

**Languages:** Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi,
Gujarati, Punjabi, Odia, Assamese, Urdu, and English. Source videos can be in
any language Whisper understands.

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
| `OPENROUTER_API_KEY` | **Yes** | Transcript clean-up, dubbing director, length fitting, Revise tab | [openrouter.ai/keys](https://openrouter.ai/keys) (add a few dollars of credit) |
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

### The older Gradio studio

The first version of the interface is still included:
`python -m ui.app` (opens on port 7860, with a temporary `gradio.live` link).

---

## 6. Using the studio

### Library (home page)

Every video dubbed so far, newest first, with its languages. Click a card to
open it; click a language chip to open that language directly. Dubs that are
running appear at the top with their progress.

### Making a new dub

1. Press **+ New dub** (top right).
2. Paste a **YouTube link**. A preview appears when the link is recognised.
3. Pick the language under **Dub into**.
4. Optional: fill in **Only part of the video** (e.g. `1:30` to `3:00`).
   Only that part is downloaded. Good for a quick first try.
5. Optional: **More options**:

   | Option | What it does |
   |---|---|
   | **Speakers** | Number of people speaking. *Detect automatically* works for most videos; setting it is the most reliable. |
   | **Content** | Keeps the right technical terms (lecture, medicine, law, …). |
   | **English words in the dub** | *None*: translate everything. *Natural*: keep the English words people really say ("phone", "meeting"). *Heavy mix*: Hinglish/Tanglish. |
   | **Translation** | *Indic + LLM polish* (default): Bodhan translates, an LLM fixes gender, tone and timing. *LLM only*: an LLM translates the whole conversation. |
   | **Songs** | *Keep original singing* (default), or *Re-sing* in the new language (experimental). |
   | **Voice engine** | Bodhan (all languages) or Sarvam (needs `SARVAM_API_KEY`). |

6. Press **Start dubbing**. A progress page shows each stage. You can leave
   it: the dub keeps running, a pill in the top bar shows its progress, and it
   opens when done.

Typical time on a 24 GB GPU: **3–5 minutes per minute of video** without lip
sync, **10–15 minutes** with lip sync. The first run is slower because models
download. Jobs run one at a time; extra jobs wait in a queue.

### Working on a dub

Opening a video shows the player and timeline on the left and five tabs on
the right.

**Player**
- **Original / Hindi / Tamil …** switches between the original and every
  language the video is dubbed in, at the same moment. Keyboard: `O`
  toggles the original.
- **⇆ Compare** shows the original and the dub side by side, split by a
  handle you can drag, playing in sync with the dub's sound. It is the
  quickest way to see the lip sync. Keyboard: `C`.
- The current dubbed line is shown over the video.
- The **timeline** has one row per speaker, with their face when it is clear
  from the video, otherwise their number. **Click** anywhere to jump there;
  **drag** across it to select the lines in that stretch.

**Selecting lines.** Drag on the timeline, or tick lines in the script
(shift-click selects a run). A bar appears with changes for just those lines,
applied straight away, with no typing of line numbers or times:
- **Voice…**: another voice for these lines. If they are all of one
  speaker's lines, that speaker gets the voice; otherwise the lines become a
  speaker of their own.
- **Speaker…**: say who really says them: an existing speaker, or a new
  person (man or woman). This fixes speakers the detection merged.
- **Delivery…**: sad, happy, news-like, storytelling, …
- **Keep original** / **Dub again**.
- **Or describe a change**: anything else, e.g. "more casual"; it is planned
  for these lines only (see Revise).

**Script tab.** Every line with its speaker, voice, original and translation.
It follows playback, and clicking a line jumps there. Hover a line for:
**▶ Orig** and **▶ Dub** to hear just that line, and **✎ Edit** to type new
wording (only that line is voiced again). Marks: **⚠ check**, transcript may
be wrong; **＋ recovered**, speech first missed; **♪ sung**.

**Voices tab.** One card per speaker. Pick another voice, press **▶** to hear
it, then **Re-voice**. Two fixes are one click away: **Actually a man/woman**
(wrong gender detected) and **Same person as…** (merge two speakers).

**Revise tab.** Describe any number of changes in one message, e.g.
*"make the intro more casual, speaker 2 is an older man with a deeper voice,
lines 4 to 6 are a different woman, don't dub the song"*. **Plan changes**
shows how each request was understood, and anything it could not do;
**Apply changes** redoes only the affected lines. With lines selected, the
request applies to those lines only.

**History tab.** Every change made to this dub, newest first, with what it
changed line by line. **▶ Watch** plays an older version in the player;
**Restore** makes it the current dub again. Restoring is itself saved as a
version, so it can be undone.

**Export tab.** The dubbed video (MP4), the soundtrack alone (M4A), and
subtitles in both languages (SRT).

**↗ Share** copies a link to a clean player for this video (`#/watch/…`):
every dubbed language, the original and Compare, without the editing tools.

**While a dub runs**, its page shows the time left and why a step may be
slow (for example, a busy GPU), then the lines as they are made: transcript,
translation, and a ▶ on each line as soon as its voice exists.

Dubs made before the web studio existed may show *"can be watched but not
edited"*. Press **Dub again**: finished steps are reused, and it becomes
editable.

All results are saved under `dubbing/cache/<video id>/`, including version
history (`versions/`); there is no database.

### Developer details

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

The **Developer → Configuration** panel shows the values in use.

---

## 9. How it works

The pipeline is a [LangGraph](https://github.com/langchain-ai/langgraph) graph
of ten stages (`dubbing/pipeline/graph.py`). Each stage is one file in
`dubbing/pipeline/nodes/`, whose opening comment lists its steps.

```
download → separate → transcribe → diarize → refine → translate
         → synthesize → align → lipsync → assemble
```

There are three graphs:

- **Full:** all ten stages, for a new video.
- **Edit:** from *translate* on, after a Revise request.
- **Re-voice:** from *synthesize* on, after a cast change.

Every stage saves its work, so a re-run redoes only lines whose input changed.

### Key ideas

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

```
Dubbing_Studio/
├── .env                      your keys (not in git)
├── scripts/setup_gpu.sh      one-time setup
└── dubbing/
    ├── pipeline/
    │   ├── graph.py          the three pipeline graphs
    │   ├── state.py          data passed between stages
    │   ├── cache.py          where each project's files live
    │   ├── nodes/            the ten stages, one file each
    │   ├── speech/           speakers, gender, pace, delivery, Bodhan ASR
    │   ├── text/             translation, director, length fitting, languages
    │   ├── voice/            voice catalogue, casting, styles, TTS, cloning
    │   ├── video/            faces, shot planning, lip sync, compositing
    │   ├── audio/            final mix
    │   ├── music/            song detection and song mode
    │   └── edits/            turning Revise requests into changes
    ├── server/               web studio backend (FastAPI): API, job queue, library
    ├── web/                  web studio page (Vue 3, no build step)
    ├── ui/                   the older Gradio studio, and helpers both studios share
    ├── tools/                command-line tools (section 7)
    ├── tests/
    ├── requirements.txt
    ├── requirements-gpu.txt
    └── .env.example
```

Created by setup and not in git: `.venv/`, `dubbing/LatentSync/`,
`dubbing/SeedVC/`, `dubbing/Separator/`, and `dubbing/cache/` (all results
and downloaded models).

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `BODHAN_TRANSLATE_KEY not set` on start | `.env` must be in the project root (next to `README.md`), not in `dubbing/`. |
| `CUDA error: no kernel image is available` | Your driver or PyTorch is too old for the GPU. Update the NVIDIA driver to 570+, then `pip install --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu128 torch torchaudio torchvision`. |
| `Sign in to confirm you're not a bot` | Set `YTDLP_COOKIES_BROWSER=firefox` (or chrome) in `.env`, or export cookies to a file and set `YTDLP_COOKIES_FILE`. |
| Log says *lip sync skipped (not enough GPU memory)* | Another program is using the GPU. Check with `nvidia-smi`. The dub is still made, without lip sync. |
| Everyone is one speaker | Set **Speakers** to the right number. Check `HF_TOKEN` and that the pyannote terms are accepted. |
| A voice has the wrong gender | Change it in the **Cast** tab, or in **Revise**: *"speaker 2 is a woman"*. |
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
