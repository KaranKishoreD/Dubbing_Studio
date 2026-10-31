"""
The dubbing engine: everything that turns a video into a dubbed video.

  graph.py   the order of the ten steps (and the shorter runs used for edits)
  state.py   the data handed from step to step: the video, its lines, settings
  cache.py   where each project's files are kept, so finished work is reused
  bodhan_gate.py  keeps calls to the Bodhan API within its rate limits

  steps/     the ten steps, one file each, in the order they run
  speech/    understanding the original audio: who speaks, gender, pace
  text/      the words: translation, the LLM "director", fitting lines to time
  voice/     the new voices: picking them, speaking the lines, cloning the actor
  video/     faces and lip sync
  audio/     mixing the final soundtrack
  music/     finding and handling songs
  edits/     turning a plain-language change request into concrete edits
"""
