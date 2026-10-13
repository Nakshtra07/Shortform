"""AutoClip Studio Presentation Engine.

Isolated sidecar package for AutoShorts 11.0 Presentation Mode 2.
Fully decoupled from web server database, sqlalchemy, and external LLM APIs.
"""
from .models import Draft, Scene, CropPoint, Packaging, PackagingCue
from .platform_strategy import PlatformStrategy, platform_strategy, list_platform_strategies
from .framing import auto_frame, window_fraction, crop_x_for_center, segment_track
from .caption_layout import width, split_screens, timed_screens
from .title_art import STYLES as TITLE_STYLES, png_bytes, artwork
from .render import render_draft

__all__ = [
    "Draft",
    "Scene",
    "CropPoint",
    "Packaging",
    "PackagingCue",
    "PlatformStrategy",
    "platform_strategy",
    "list_platform_strategies",
    "auto_frame",
    "window_fraction",
    "crop_x_for_center",
    "segment_track",
    "width",
    "split_screens",
    "timed_screens",
    "TITLE_STYLES",
    "png_bytes",
    "artwork",
    "render_draft",
]
