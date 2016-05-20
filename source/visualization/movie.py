import moviepy.editor
import numpy as np
import gizeh

from source.io.path import default_path
from source.struct.agent import Agent


def make_frame(t, agent: Agent = None):
    surface = gizeh.Surface(width=400, height=400)

    return surface.get_npimage()


clip = moviepy.editor.VideoClip(make_frame, duration=10)
clip.write_gif(default_path('crowd.gif', 'documentation', 'animations'),
               fps=30)
