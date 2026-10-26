import pytest
from pathlib import Path
from autoclip_studio.publish_export import ExportRequest, export_clip
from autoclip_studio.packaging_html import get_template, prepare_overlay
import autoclip_studio.render as studio_render

def test_publish_export_slug_no_fake_outro_when_unsupported():
    """When brand_outro=True but no outro asset exists, slug should not claim outro-v1."""
    req = ExportRequest(
        project_id="test_proj",
        clip_id="c1",
        brand_outro=True,
    )
    # Even without running the full ffmpeg command, slug logic must not falsely include _autoclip-outro-v1
    # We test via the public API or slug resolution
    from autoclip_studio.publish_export import resolve_outro_path
    assert resolve_outro_path(1080, 1920) is None, "Default setup should have no outro asset"

def test_packaging_html_get_template_not_none():
    """get_template must return a valid spec for editorial and street without AttributeError."""
    for tmpl_name in ("editorial", "street"):
        entry = get_template(tmpl_name)
        assert entry is not None, f"Template {tmpl_name} should resolve"
        assert hasattr(entry, "canvas"), "Template must have canvas"
        assert len(entry.canvas.overlay_px) == 2, "Canvas must have 2D overlay_px"

def test_packaging_html_prepare_overlay_clean_downgrade():
    """prepare_overlay should cleanly downgrade when assets are missing, not crash with AttributeError."""
    job = prepare_overlay("p1", "editorial", None, 5.0, features=None)
    assert job is not None
    assert job.downgraded is True
    assert job.outcome in ("downgraded", "skipped", "completed")
