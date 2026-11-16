"""Editor Agent: builds the editing timeline with deterministic rules (no LLM call).

Timing is driven by the MEASURED narration audio, never by word-count estimates alone.
Spoken audio is never trimmed: scenes get exactly their narration length plus a pause.
"""
from __future__ import annotations

import json
import re
import zlib
from pathlib import Path

from studio.agents.base import Agent
from studio.schemas import AudioClip, Caption, EditingTimeline, MediaAsset, Storyboard, TimelineClip
from studio.utils import split_sentences, word_count

SCENE_PAUSE_S = 0.35        # default breathing room after each scene's narration
MAX_EXTRA_PAUSE_S = 1.5     # most that may be added per scene to reach the minimum duration
CAPTION_WORDS = 9


def caption_chunks(narration: str) -> list[str]:
    """Split narration into short caption lines, breaking at sentence ends where possible."""
    chunks = []
    for sentence in split_sentences(narration):
        words = sentence.split()
        for start in range(0, len(words), CAPTION_WORDS):
            chunk = words[start:start + CAPTION_WORDS]
            if len(chunk) <= 2 and chunks and start:          # avoid orphan fragments
                chunks[-1] += " " + " ".join(chunk)
            else:
                chunks.append(" ".join(chunk))
    return chunks


def pick_track(music_dir: Path | str, tone: str, seed: str) -> dict | None:
    """Choose a background track from the library manifest (`tracks.json`), or None.

    The track whose mood tags share the most words with the video's tone wins; ties (and a tone that
    matches nothing) are settled by a hash of `seed`, so one project always gets the same track
    while different projects are spread over the library. Entries without a licence are skipped.
    """
    folder = Path(music_dir)
    try:
        entries = json.loads((folder / "tracks.json").read_text(encoding="utf-8"))["tracks"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    usable = [t for t in entries if isinstance(t, dict) and str(t.get("license", "")).strip()
              and (folder / str(t.get("file", ""))).is_file()]
    if not usable:
        return None
    words = set(re.findall(r"[a-z]+", tone.lower()))
    score = {t["file"]: len(words & {m.lower() for m in t.get("moods", [])}) for t in usable}
    best = [t for t in usable if score[t["file"]] == max(score.values())]
    best.sort(key=lambda t: t["file"])
    track = dict(best[zlib.crc32(seed.encode()) % len(best)])
    track["path"] = str(folder / track["file"])
    return track


class Editor(Agent):
    name = "editor"

    def build_timeline(self, storyboard: Storyboard, visuals: dict[str, MediaAsset],
                       narration: dict[str, MediaAsset]) -> EditingTimeline:
        """visuals: {shot_id: image or video asset}; narration: {scene_id: audio asset}."""
        s = self.ctx.settings
        fps, notes = s.fps, []
        speech_total = sum(narration[scene.id].duration_s for scene in storyboard.scenes)
        pause = SCENE_PAUSE_S
        minimum, maximum = s.duration_bounds
        base_total = speech_total + pause * len(storyboard.scenes)
        if base_total < minimum:
            extra = min((minimum - base_total) / len(storyboard.scenes), MAX_EXTRA_PAUSE_S)
            pause += extra
            notes.append(f"Narration is {speech_total:.1f}s; added {extra:.2f}s pause per scene to "
                         f"approach the {s.target_duration_s:.0f}s target.")
        if speech_total > maximum:
            notes.append(f"Narration ({speech_total:.1f}s) exceeds the accepted {maximum:.0f}s; it is not cut. "
                         "Shorten the script to fix the duration.")

        clips, audio, frame_cursor = [], [], 0
        for scene in storyboard.scenes:
            sound = narration[scene.id]
            scene_start_frame = frame_cursor
            scene_start = frame_cursor / fps
            total_words = max(sum(word_count(shot.narration) for shot in scene.shots), 1)
            spoken_words = 0
            for position, shot in enumerate(scene.shots):
                last = position == len(scene.shots) - 1
                speech_start = scene_start + sound.duration_s * spoken_words / total_words
                spoken_words += word_count(shot.narration)
                speech_end = scene_start + sound.duration_s * spoken_words / total_words
                end_frame = round((speech_end + (pause if last else 0)) * fps)
                end_frame = max(end_frame, frame_cursor + max(fps // 2, 1))     # at least 0.5s on screen
                asset = visuals[shot.id]
                clip = TimelineClip(
                    shot_id=shot.id, scene_id=scene.id, asset_path=asset.path,
                    asset_kind="video" if asset.kind == "video" else "image",
                    start_s=frame_cursor / fps, duration_s=(end_frame - frame_cursor) / fps,
                    camera_move=shot.camera_move, fade_in=position == 0, fade_out=last)
                if s.captions_enabled:
                    chunks = caption_chunks(shot.narration)
                    words, done = max(word_count(shot.narration), 1), 0
                    for chunk in chunks:
                        start = speech_start + (speech_end - speech_start) * done / words
                        done += word_count(chunk)
                        end = speech_start + (speech_end - speech_start) * done / words
                        clip.captions.append(Caption(text=chunk, start_s=round(start, 3), end_s=round(end, 3)))
                clips.append(clip)
                frame_cursor = end_frame
            audio.append(AudioClip(scene_id=scene.id, path=sound.path, start_s=scene_start,
                                   slot_s=(frame_cursor - scene_start_frame) / fps, audio_s=sound.duration_s))

        music, credit = "", {}
        if s.enable_music and s.music_path:                       # one file forced by the user
            if Path(s.music_path).is_file():
                music, credit = s.music_path, {"music_license": s.music_license}
            else:
                notes.append(f"MUSIC_PATH '{s.music_path}' was not found; the video has no music.")
        elif s.enable_music:                                      # a track from the library
            tone = str(self.ctx.store.recall(self.ctx.project_id, "tone", ""))
            track = pick_track(s.music_dir, tone, self.ctx.project_id)
            if track:
                music = track["path"]
                credit = {"music_title": f"{track.get('title', track['file'])} by {track.get('author', 'unknown')}",
                          "music_license": track["license"], "music_source": track.get("source", "")}
            else:
                notes.append(f"No licensed track found in '{s.music_dir}'; the video has no music.")
        timeline = EditingTimeline(
            project_id=self.ctx.project_id, parent_id=storyboard.id, width=s.width, height=s.height, fps=fps,
            clips=clips, audio=audio, music_path=music, music_volume=s.music_volume if music else 0.0,
            total_duration_s=frame_cursor / fps, notes=notes, status="ready", **credit)
        self.note("build_timeline", f"{len(clips)} clips, {len(audio)} narration segments, "
                                    f"{timeline.total_duration_s:.1f}s")
        return timeline
