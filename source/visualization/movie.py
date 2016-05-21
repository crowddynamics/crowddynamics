import gizeh
from moviepy.editor import VideoClip

from source.io.path import default_path


def make_frame(t):
    surface = gizeh.Surface(width=800, height=800, bg_color=(1, 1, 1))
    circle = gizeh.circle(100, xy=(400, 400), fill=(1, 0, 0))
    circle.draw(surface)
    return surface.get_npimage()


clip = VideoClip(make_frame, duration=1)
clip.write_videofile(default_path('crowd.mp4', 'documentation', 'animations'),
                     fps=24)
