# AI Video Studio

A local-first, multi-agent production studio that turns a topic into a finished
YouTube-style video: plan, research, script, storyboard, images, narration, FFmpeg
assembly, quality and compliance review, SEO metadata and a reviewed thumbnail.

Default format: **"Your Life as Every Rank in [TOPIC]"** (10 minutes, 16:9, 1080p, English).
Nothing is uploaded or published; every result stays on your disk.

Full design documentation: [docs/SYSTEM_DOCUMENTATION.md](docs/SYSTEM_DOCUMENTATION.md).

## Main features

- 13 specialised agents (Director, Research, Story Writer, Scene Director, Prompt Engineer,
  Prompt Judge, Visual Generation, Visual Judge, Voice Director, Editor, Final QA, SEO & Growth,
  Compliance) coordinated by a custom Python workflow engine. No agent framework.
- Deterministic software guards (script, timeline, prompt, media, audio, final video) separate
  from LLM judges.
- Feedback loops with bounded retries: only the rejected artifact is redone; approved work is kept.
- Final QA sends a video that is too long back to the Scene Director, who removes whole sentences
  (no rewording); images are kept and only the changed scenes are re-recorded and re-rendered.
- Background music: the Editor picks a public-domain (CC0) track from `assets/music/` that fits the
  video's tone and mixes it quietly under the narration (`ENABLE_MUSIC`, `MUSIC_VOLUME`).
- Complete thumbnail branch: SEO brief, prompt, prompt judge, generation, SEO review, visual
  judge, Director approval, with its own retry counter and its own, stronger model
  (`THUMBNAIL_MODEL`). Thumbnail text is drawn locally.
- One SQLite file as unified memory (tasks, agent runs, versions, assets, evaluations, costs,
  decisions). Media files live on disk and are referenced by path.
- One OpenRouter client for every model call (chat, vision, images, video jobs, speech).
- Three modes: `dry-run` (no API calls, labelled placeholders), `preview` (short paid sample),
  `produce` (full paid production), with budget confirmation and a hard budget stop.
- Checkpointing and resume, human override for escalated tasks, CLI, and a dashboard that
  starts, queues, monitors and recovers productions through a local background worker.

## Requirements

- Python 3.11 or newer
- FFmpeg and ffprobe on your PATH (`brew install ffmpeg`, `apt install ffmpeg`, ...)
- An OpenRouter API key, only for the live modes

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt       # everything: app, dashboard and tests
cp .env.example .env                 # then put your key in OPENROUTER_API_KEY
```

`requirements.txt` lists the same packages as `pyproject.toml`. Use
`pip install -e ".[dev,dashboard]"` instead if you also want the `studio` command.

Every command below is shown as `python run.py ...`, which works straight from the project
folder. After `pip install -e .` the same commands are also available as `studio ...` or
`python -m studio ...`.

> macOS note: if `python -m studio` reports "No module named studio" after an editable
> install, use `python run.py` (Python 3.13 skips install files that macOS has marked hidden).

## Configuration

All settings are environment variables, documented in [.env.example](.env.example): model ids,
duration and resolution, voice, feature switches (images, video, TTS), judge thresholds, retry
limits, paths, timeouts, polling, budget and media-request cap.

```bash
python run.py check            # validates the configuration and looks for ffmpeg
python run.py check --models   # also verifies model ids, voice and options (free, no generation)
```

Model ids change over time. `check --models` uses OpenRouter's public catalog and tells you
exactly which setting is wrong; the same check runs automatically before every live run.

## CLI

```bash
python run.py dry-run "Cybersecurity"                 # full pipeline, no API calls, no cost
python run.py preview "Cybersecurity" --budget 2      # ~60 s live sample (asks for confirmation)
python run.py produce "Cybersecurity" --budget 25     # full 10-minute live production
python run.py new "Nursing" --mode preview            # create without running
python run.py run <project-id>                        # run a created project
python run.py status <project-id>                     # stage, task states, errors, cost
python run.py watch [project-id]                      # live log: what each agent is doing and saying
python run.py report <project-id>                     # cost, evaluation, decision summaries
python run.py collect <project-id>                    # refresh output/<id>/ from local files (no API call)
python run.py resume <project-id>                     # continue after a failure or budget stop
python run.py resume <project-id> --budget 40         # raise the budget and continue
python run.py resume <project-id> --retry-escalated   # give NEEDS_REVIEW tasks fresh attempts
python run.py override <project-id> <task> --reason "..."   # record a human decision
python run.py list
python run.py queue                                   # jobs queued from the dashboard, worker state
python run.py worker                                  # run the queued jobs (the dashboard starts this itself)
```

Useful options: `--sources DIR` (local `.txt`/`.md` research files), `--title`, `--duration
SECONDS`, `--enable-video` (AI video clips, off by default), `--yes` (skip the confirmation
prompt), `-v` before the command (print the run log).

Live runs print the models, budget and request cap and wait for you to type `yes`. The budget
is checked before every request; when it is reached the run stops and can be resumed.

## Dashboard

Start, monitor and recover productions from the browser. It uses the same pipeline, database
and budget rules as the CLI.

```bash
pip install -r requirements.txt        # once; includes Streamlit 1.50 or newer
streamlit run dashboard/app.py         # run from the project folder, then open http://localhost:8501
```

The sidebar switches between five views and selects the project.

| View | What it does |
| --- | --- |
| **Monitor** | Live status of the selected project: stage, active agent, tasks and progress bar, elapsed time, recorded cost, unknown-cost requests, images / voiceovers / clips, retries and rejections, latest result, the reason for any block. Agent table (all 13 agents plus the software guards; agents not yet invoked show Pending) with per-agent details: tasks, attempts, time, model calls, evaluations, feedback, errors, inputs and outputs. Evaluation messages, event feed, run log. Refreshes every 3 s while a production is running. |
| **Media** | Rendered video, thumbnail, images, narration audio, clips, each labelled APPROVED or PRELIMINARY. The video is shown as soon as it is rendered. |
| **New production** | Topic, automatic title, mode, budget, optional AI video clips, estimated scope, confirmation, **Start Production**. |
| **Batch & queue** | Several topics (one per line) become independent productions, queued and run one at a time. Shows running, queued, completed and failed/blocked jobs; lets you cancel a queued job or stop the running one after its current task. |
| **Recovery** | Full failure reason for a failed or needs-review project and a recovery action that runs only after you tick the confirmation. |

### Start a demo production

1. Open **New production** and type a topic, for example `Cybersecurity`. The title becomes
   "Your Life as Every Rank in Cybersecurity".
2. Choose **Demo mode (60 seconds, paid)**. (**Full production mode** is the 600-second video;
   **Offline test** runs the whole pipeline with sample media, free, to try the dashboard.)
3. Set the maximum budget, read the estimated scope, tick the confirmation and click
   **Start Production**. The Monitor view of the new project opens automatically.

### Start a batch

1. Open **Batch & queue** and enter one topic per line.
2. Choose the mode and the budget **per production**, tick the confirmation (it states the
   maximum total) and click **Queue batch**.
3. The queue below shows what is running and what is waiting. Pick any project in the sidebar
   to monitor it.

### How productions run

- **Separate worker process.** Clicking Start writes a job to SQLite and starts a detached
  worker (`python -m studio worker`). The page never runs the pipeline, so it stays
  responsive; you can switch views, reload or close the dashboard and the production
  continues. The worker runs the queue one job at a time and exits when it is empty.
- **No duplicates.** A repeated click or a Streamlit rerun returns the same job; a project can
  have only one queued or running job (enforced by SQLite); only one worker can be active; and
  a project is never run by two processes at once, so launches cannot share a budget.
- **Budgets and confirmation are enforced in the core**, not in the page: a paid job is
  refused without an explicit confirmation and an API key, and the orchestrator checks the
  project's own budget before every request.
- **Stopping.** "Stop the running production" lets the current task finish, then stops;
  completed tasks and approved assets are kept. If the worker process is killed or the
  computer shuts down, the job is marked `interrupted` the next time the dashboard or
  `python run.py queue` runs, and the project can be resumed from the Recovery view.
- **Recovery** uses the existing resume and escalation logic. Rejected work is never approved
  automatically: accepting it needs a written reason and is stored as a human decision.
  Compliance blocks cannot be accepted from the dashboard (CLI `override` only).
- **The API key is never shown.** The page only says whether a key is configured. Worker
  output goes to `data/worker.log`; project logs stay redacted.
- **Local only.** `.streamlit/config.toml` binds the server to `localhost`.

Code layout: `dashboard/app.py` (page), `dashboard/monitor.py` (read-only queries, SQLite
opened read-only), `dashboard/launcher.py` (actions). Queue and worker logic lives in the core
package (`src/studio/orchestration/jobs.py`), so another interface can reuse it.

## Output locations

`output/<project-id>/` is filled as soon as there is something to show, not only at the end.
Top-level media and reports are written only by the export task, after every approval.

| Path | Content |
| --- | --- |
| `output/<project-id>/STATUS.txt` | `APPROVED` or `INCOMPLETE - NOT APPROVED`, with the tasks still open and why |
| `output/<project-id>/preliminary/UNAPPROVED_video.mp4` | The rendered video before final review; removed once the production is approved |
| `output/<project-id>/assets/` | `images/`, `audio/`, `references/`, `video/` (approved assets) and `thumbnails/` (all candidates, status in the file name) |
| `output/<project-id>/artifacts/` | Latest plan, script (`script.txt`), storyboard, prompts, timeline, reports as JSON |
| `output/<project-id>/video.mp4` | Final video (H.264 + AAC, burned-in captions) |
| `output/<project-id>/thumbnail.png` | Approved thumbnail, 1280x720 |
| `output/<project-id>/metadata.json` | Title options, description, tags, keywords, disclosure |
| `output/<project-id>/description.txt` | YouTube description |
| `output/<project-id>/captions.srt` | Captions as a separate file |
| `output/<project-id>/quality_report.json` | Guard and judge results, revisions, limitations |
| `output/<project-id>/compliance_report.json` | Compliance checklist and human decisions |
| `output/<project-id>/production_report.json` | Tasks, costs, outputs |
| `data/projects/<project-id>/` | Working files: images, audio, thumbnail candidates, render, logs |
| `data/studio.db` | SQLite memory for all projects |

## Testing

```bash
python -m pytest
```

112 tests, no network access and no paid calls (OpenRouter is mocked; a fixture fails any test
that tries to reach the network). The end-to-end and rendering tests need FFmpeg and are
skipped without it.

## Troubleshooting

- **Requests hang for about a minute each**: some networks advertise a broken IPv6 route.
  `FORCE_IPV4=true` (the default) makes the client connect over IPv4 only.
- **Everything Python is very slow, or `No module named studio`**: this happens when the
  project folder is inside an iCloud-synced location such as `~/Documents`, because iCloud
  syncs the thousands of files in `.venv`. Keep the virtual environment outside the synced
  folder (`python3 -m venv ~/venvs/studio`) or move the project out of iCloud.
- **`model ... is not in the OpenRouter model catalog`**: update the id in `.env`; run
  `python run.py check --models` until it reports all models usable.
- **Final Video Guard: duration outside the accepted range** (live runs): the real narration
  was longer or shorter than estimated. Nothing is lost: the render is in
  `output/<id>/preliminary/`. Adjust `WORDS_PER_MINUTE` to the voice's real pace for future
  runs, widen the range, or accept the result with `override <project> final_qa --reason "..."`,
  then `resume`. Retrying Final QA re-uses the stored continuity review and costs nothing.

## Known limitations

- **Live modes have not been run against paid endpoints during development.** Request and
  response formats follow the OpenRouter documentation and are covered by mocked tests, and the
  default model ids were verified against the live catalog, but run `preview` with a small
  budget before a full production.
- Scenes are rendered one after another and queued productions run one at a time (no parallel
  generation). A full production makes
  roughly 50-75 image requests and 15-30 speech requests plus about 100-150 LLM calls.
- Narration timing inside a scene is proportional to word counts, so caption timing is
  approximate (scene boundaries are exact).
- The final duration follows the real narration length and is never stretched or trimmed.
  Full productions must land within `DURATION_TOLERANCE` of the target; previews are accepted
  between `PREVIEW_MIN_DURATION_S` and `PREVIEW_MAX_DURATION_S` (50-80 s by default). Outside
  that range the Final Video Guard escalates to a human.
- Character consistency depends on the image model. Reference images are passed only when the
  model accepts them; otherwise the limitation is recorded in the reports.
- AI video clips are optional, capped (`MAX_VIDEO_CLIPS`), checked by the Media Guard only and
  not reviewed by the Visual Judge.
- Speech cost is looked up after the request; when OpenRouter does not return it, it is
  reported as unknown and is not part of the budget arithmetic (the media request cap still applies).
- Transcription comparison runs only when `STT_MODEL` is set.
- Compliance checks are screening aids, not legal advice. Several items always require human
  review (fact checking, synthetic-media disclosure).
- The dashboard's timed refresh and the improved thumbnail review (below) have been tested
  with sample data only, not yet in a live run.
- Thumbnail title text is drawn by software in a banner along the bottom edge; its position
  and style are fixed.
- Transitions are fades at scene boundaries and cuts inside scenes; sound effects are not implemented.
