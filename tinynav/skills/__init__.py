"""
tinynav/skills — Blocking CLI skills for robot task composition.

Each skill is a standalone executable that blocks until its action completes.
Compose them with shell scripts or let agents write new ones on the fly.

Available skills:
  go      – Navigate to a POI by id or name
  lookat  – Rotate to face a map-frame coordinate
  wait    – Sleep for N seconds
  photo   – Capture and save a keyframe image
"""
