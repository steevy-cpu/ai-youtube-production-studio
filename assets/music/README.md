# Background music library

Tracks the Editor may place quietly under the narration. Every file here was published by its
author on OpenGameArt.org under **CC0 1.0** (public domain dedication): it may be copied, modified
and used commercially without permission or credit. The licence shown on each source page was
checked on 2026-11-15; `tracks.json` records title, author, licence and source for every file, and
that record is copied into each production's asset list and compliance report.

| File | Title | Author | Length | Mood tags | Source |
| --- | --- | --- | --- | --- | --- |
| `chill_lofi_inspired.mp3` | Chill lofi inspired | isaiah658 | 123s | chill, calm, relaxed, lofi | [page](https://opengameart.org/content/chill-lofi-inspired) |
| `apple_cider.mp3` | Apple Cider | SeventhSeeker | 200s | cozy, warm, gentle, friendly | [page](https://opengameart.org/content/apple-cider) |
| `a_day_in_night_out.mp3` | A Day In, Night Out | Umplix | 77s | fun, playful, upbeat, lofi | [page](https://opengameart.org/content/a-day-in-night-out) |
| `next_to_you.mp3` | Next to You | SdiviHall | 84s | emotional, inspiring, aspirational, heartfelt | [page](https://opengameart.org/content/next-to-you) |
| `bossa_nova.mp3` | Bossa Nova | looneybits | 60s | light, quirky, comedic, funny | [page](https://opengameart.org/content/bossa-nova) |

The files were converted to 128 kbps MP3 and loudness-normalised to -18 LUFS so that `MUSIC_VOLUME`
behaves the same for every track. Nothing else was changed.

## Adding a track

1. Only add music whose licence allows redistribution and commercial use, and keep the source link.
2. Put the file in this folder and add an entry to `tracks.json` (same fields as the others).
   `moods` are the words matched against the video's tone when a track is chosen.

A track is only used when it has a `tracks.json` entry with a licence. Set `ENABLE_MUSIC=false` to
turn background music off, or `MUSIC_PATH` + `MUSIC_LICENSE` to force one specific file.
