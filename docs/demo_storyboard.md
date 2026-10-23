# Demo video storyboard

A shot-by-shot script for a **75-second** screen recording of the
LaneIQ live demo. Target: a recruiter or hiring manager who skims.
The goal of every shot is "this is a real, working system."

## Before you record

1. Start the backend: `uv run uvicorn backend.app.main:app --port 8000`
   (drop `--reload` for recording — avoids a mid-take restart).
2. Start the frontend: `cd frontend && npm run dev`.
3. Open `http://localhost:5173` in a clean browser window. Hide
   bookmarks bar, use a 1280×800-ish window so text is legible when
   the video is downscaled to a GIF.
4. Do one throwaway run end-to-end so SUMO is warm and you know the
   timing.
5. Recommended tool: macOS QuickTime screen recording, or Loom.
   Export → trim → convert to GIF with `ffmpeg` or [gifski] for the
   README hero (keep the GIF under ~8 MB; ~12 fps is fine).

## The 75-second cut

| Time | Shot | What you do | What to say (voiceover or caption) |
| --- | --- | --- | --- |
| 0:00–0:08 | **Title + dashboard** | Dashboard already open, nothing running. Slow cursor move across the panels. | "LaneIQ — a reinforcement-learning agent that drives a highway, built end-to-end: SUMO simulator, two RL algorithm families, a live dashboard." |
| 0:08–0:18 | **Start the canonical agent** | Quick-pick **Conservative** (`ppo_v12`), switch density to **high**, click **Start live**. | "This is a PPO agent I trained on a 6-component reward. It decides when to change lanes; SUMO's lane-change assist refuses any change that would cut into another car. In heavy traffic it finishes about a second per kilometre faster than the rule-based drivers, paired over 20 seeds." |
| 0:18–0:30 | **Watch it drive** | Let the canvas run. Point cursor at the gold EGO. Point at the DecisionTicker as the chip changes. | "The gold car is the agent. Up top, it narrates every decision in plain English: the lane change the car actually made, streamed at 10 Hz." |
| 0:30–0:40 | **Metrics** | Point at the MetricsPanel — speed counter, the climbing cumulative-return curve, the reward-component chart. | "Live metrics: episode return climbing, the six reward components, current speed and lane." |
| 0:40–0:42 | **Stop** | Click **Stop**. | (pause) |
| 0:42–0:55 | **The gauntlet** | Quick-pick **Gauntlet** (`ppo_v10`; sets agent + density together). Click **Start live**. Let it run. | "The gauntlet starts the car stuck behind a wall of slow traffic. Watch it pick a gap, change lanes and get out." |
| 0:55–1:05 | **Aggressive** | Stop. Quick-pick **Aggressive** (the rule-based overtaker on the 10 km rush road). Start. | "For contrast, a hand-written overtaker on a faster road: it weaves past 33 m/s traffic at about 42 m/s, and the same lane-change assist keeps it collision-free." |
| 1:05–1:15 | **The evidence** | Stop. Scroll down. Expand **"How it learned"** then **"How it uses lanes"**. | "Everything's backed by reproducible evaluation: 20 seeds per agent and density, ranked by finish time. A big part of the work was finding why the early numbers were wrong. It's all in the repo." |

## Alternative 30-second cut (for the README hero GIF)

The README GIF should be short and loop cleanly. Use just the
gauntlet payoff:

| Time | Shot |
| --- | --- |
| 0:00–0:03 | Dashboard, click **Gauntlet** quick-pick |
| 0:03–0:06 | Click **Start live** |
| 0:06–0:24 | Ego spawns stuck in lane 0, then changes out and accelerates — the DecisionTicker flashing CHANGE LEFT as it does |
| 0:24–0:30 | MetricsPanel return curve climbing |

Loops cleanly because it starts and ends on a calm frame.

## Tips

- **Turn on "loop episodes"** for the long cut if you want continuous
  motion without clicking — but turn it OFF for the gauntlet contrast
  shot so the agent swap is clean.
- The DecisionTicker flash is subtle on a downscaled GIF — consider a
  cursor highlight or a zoom-in for shots 0:18–0:30.
- If SUMO hiccups mid-take, just re-run — episodes are ~30 seconds.
- Keep the browser zoom at 100%. The canvas is responsive but text
  panels read best un-zoomed.

[gifski]: https://gif.ski/
