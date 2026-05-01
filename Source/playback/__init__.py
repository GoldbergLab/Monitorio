"""Random video playback utilities for Monitorio experiments.

Plays a randomly-chosen video from a configured set, with an
exponentially-distributed inter-video interval, on a chosen monitor in
fullscreen, and logs each playback's start time to a CSV. Designed to
be triggered alongside continuous DAQ/Intan recording so that the
recorded photodiode signals + this script's wall-clock log together
let the analyst extract precise frame-by-frame neural-spike alignment
later.
"""
