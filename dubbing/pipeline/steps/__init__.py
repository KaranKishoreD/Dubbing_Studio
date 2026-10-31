"""
The ten steps of a dub, one file each, in the order they run:

  download → separate → transcribe → diarize → refine →
  translate → synthesize → align → lipsync → assemble

Each step reads the shared state (pipeline/state.py), does one job, saves
its results under cache/<project>/, and returns what it changed.
"""
