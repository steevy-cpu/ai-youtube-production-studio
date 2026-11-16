"""Deterministic FFmpeg renderer: executes an EditingTimeline and writes the final MP4.

Steps: (1) one silent H.264 clip per shot (pan/zoom still or fitted video clip, fades,
caption overlays drawn with Pillow), (2) lossless concat of the clips, (3) narration track
assembled from per-scene audio padded to its slot, (4) optional music mix, (5) final mux.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from studio.rendering.ffmpeg import run_ffmpeg
from studio.schemas import Caption, EditingTimeline, TimelineClip
from studio.services.placeholders import load_font

ZOOM = 0.12     # how far stills zoom / pan over a shot
ENCODE = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
          "-video_track_timescale", "90000", "-an"]


def caption_png(text: str, width: int, height: int, path: Path, font_path: str = "") -> None:
    """Draw one caption (max two lines, white with black outline) on a transparent frame."""
    size = max(height // 22, 12)
    font = load_font(size, font_path)
    lines = textwrap.wrap(text, width=max(int(width / (size * 0.55)), 12))[:3]
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    block = "\n".join(lines)
    box = draw.multiline_textbbox((0, 0), block, font=font, align="center", stroke_width=max(size // 10, 1))
    x = (width - (box[2] - box[0])) / 2 - box[0]
    y = height - (box[3] - box[1]) - height // 14 - box[1]
    draw.multiline_text((x, y), block, font=font, fill=(255, 255, 255, 255), align="center",
                        stroke_width=max(size // 10, 1), stroke_fill=(0, 0, 0, 255))
    img.save(path, "PNG")


def write_srt(captions: list[Caption], path: Path) -> None:
    def stamp(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    lines = [f"{n}\n{stamp(c.start_s)} --> {stamp(c.end_s)}\n{c.text}\n"
             for n, c in enumerate(captions, start=1)]
    path.write_text("\n".join(lines), encoding="utf-8")


def _motion(move: str, frames: int, width: int, height: int, fps: int) -> str:
    n = max(frames - 1, 1)
    centre = "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    expr = {
        "zoom_in": f"z='1+{ZOOM}*on/{n}':{centre}",
        "zoom_out": f"z='{1 + ZOOM}-{ZOOM}*on/{n}':{centre}",
        "pan_left": f"z='{1 + ZOOM}':x='(iw-iw/zoom)*(1-on/{n})':y='ih/2-(ih/zoom/2)'",
        "pan_right": f"z='{1 + ZOOM}':x='(iw-iw/zoom)*on/{n}':y='ih/2-(ih/zoom/2)'",
        "static": f"z='1':{centre}",
    }[move]
    return f"zoompan={expr}:d={frames}:s={width}x{height}:fps={fps}"


def render_clip(clip: TimelineClip, timeline: EditingTimeline, out: Path, work: Path, font_path: str = "") -> None:
    w, h, fps = timeline.width, timeline.height, timeline.fps
    frames = max(round(clip.duration_s * fps), 1)
    duration = frames / fps
    inputs = ["-i", clip.asset_path]
    if clip.asset_kind == "image":
        chain = (f"[0:v]scale={w * 2}:{h * 2}:force_original_aspect_ratio=increase,crop={w * 2}:{h * 2},"
                 f"{_motion(clip.camera_move, frames, w, h, fps)}")
    else:   # generated clip: fit the frame, then hold the last frame if it is shorter than the shot
        chain = (f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps},"
                 f"tpad=stop_mode=clone:stop_duration={duration:.3f},trim=duration={duration:.3f},"
                 "setpts=PTS-STARTPTS")
    fade = timeline.transition_s
    if duration > fade * 3:
        if clip.fade_in:
            chain += f",fade=t=in:st=0:d={fade}"
        if clip.fade_out:
            chain += f",fade=t=out:st={duration - fade:.3f}:d={fade}"
    chain += ",format=yuv420p[v0]"
    filters = [chain]
    for number, caption in enumerate(clip.captions, start=1):
        png = work / f"{out.stem}_cap{number}.png"
        caption_png(caption.text, w, h, png, font_path)
        inputs += ["-i", str(png)]
        start = max(caption.start_s - clip.start_s, 0.0)
        end = min(caption.end_s - clip.start_s, duration)
        filters.append(f"[v{number - 1}][{number}:v]overlay=0:0:enable='between(t,{start:.3f},{end:.3f})'"
                       f"[v{number}]")
    run_ffmpeg([*inputs, "-filter_complex", ";".join(filters), "-map", f"[v{len(clip.captions)}]",
                "-frames:v", str(frames), "-r", str(fps), *ENCODE, str(out)])


def _concat(files: list[Path], out: Path, list_file: Path, codec: list[str]) -> None:
    # Entries are bare file names next to the list file, so no shell or path quoting issues arise.
    list_file.write_text("".join(f"file '{f.name}'\n" for f in files), encoding="utf-8")
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_file), *codec, str(out)])


def render(timeline: EditingTimeline, work_dir: Path, out_path: Path, font_path: str = "", log=None) -> Path:
    """Render the timeline to `out_path` (MP4, H.264 + AAC). Returns the output path."""
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clip_files = []
    for number, clip in enumerate(timeline.clips, start=1):
        clip_file = work_dir / f"clip_{number:04d}.mp4"
        render_clip(clip, timeline, clip_file, work_dir, font_path)
        clip_files.append(clip_file)
        if log and number % 10 == 0:
            log.info(f"renderer: {number}/{len(timeline.clips)} clips rendered")
    video = work_dir / "video_only.mp4"
    _concat(clip_files, video, work_dir / "clips.txt", ["-c", "copy"])

    audio_files = []
    for number, clip in enumerate(timeline.audio, start=1):
        audio_file = work_dir / f"scene_audio_{number:04d}.wav"
        # apad + -t pads each narration with silence to its slot; speech itself is never trimmed
        # because the Editor always makes the slot at least as long as the measured audio.
        run_ffmpeg(["-i", clip.path, "-af", "apad", "-t", f"{clip.slot_s:.4f}", "-ar", "44100", "-ac", "2",
                    "-c:a", "pcm_s16le", str(audio_file)])
        audio_files.append(audio_file)
    narration = work_dir / "narration.wav"
    _concat(audio_files, narration, work_dir / "audio.txt", ["-c", "copy"])

    total = f"{timeline.total_duration_s:.3f}"
    args = ["-i", str(video), "-i", str(narration)]
    if timeline.music_path:
        args += ["-stream_loop", "-1", "-i", timeline.music_path, "-filter_complex",
                 # quiet bed under the voice, faded in and out so it never starts or stops abruptly
                 f"[2:a]volume={timeline.music_volume},afade=t=in:d=1.5,"
                 f"afade=t=out:st={max(timeline.total_duration_s - 3, 0):.3f}:d=3[m];"
                 "[1:a][m]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]",
                 "-map", "0:v", "-map", "[a]"]
    else:
        args += ["-map", "0:v", "-map", "1:a"]
    run_ffmpeg([*args, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", total,
                "-movflags", "+faststart", str(out_path)])
    write_srt([c for clip in timeline.clips for c in clip.captions], out_path.with_suffix(".srt"))
    return out_path
