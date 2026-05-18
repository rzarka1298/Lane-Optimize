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
| 0:08–0:18 | **Start the canonical agent** | Quick-pick **Conservative**, density **high**, click **Start live**. | "This is a PPO agent I trained from a 6-component reward. It's the only one of five agents with a zero-percent collision rate at every traffic density." |
| 0:18–0:30 | **Watch it drive** | Let the canvas run. Point cursor at the gold EGO. Point at the DecisionTicker as the chip changes. | "The gold car is the agent. Up top, it narrates every decision in plain English — this is the policy's actual chosen action, streamed at 10 Hz." |
| 0:30–0:40 | **Metrics** | Point at the MetricsPanel — speed counter, the climbing cumulative-return curve, the reward-component chart. | "Live metrics: episode return climbing, the six reward components, current speed and lane." |
| 0:40–0:42 | **Stop** | Click **Stop**. | (pause) |
| 0:42–0:58 | **The gauntlet — the payoff** | Quick-pick **Gauntlet** (it sets agent + density together). Click **Start live**. Let it run. | "Now the interesting part. The 'gauntlet' scenario forces the car to start stuck behind a wall of slow traffic. A reward change alone couldn't make an agent overtake — I had to redesign the environment. Watch it escape the slow lane and accelerate away." |
| 0:58–1:05 | **Contrast** | Stop. Switch agent dropdown to **Conservative** (keep density gauntlet). Start. It sits in lane 0. | "Same scenario, the agent that wasn't trained for it: it just gives up and follows the slow wall. 25 metres per second versus 33." |
| 1:05–1:15 | **The evidence** | Stop. Scroll down. Expand **"How it learned"** then **"Overtaking"** tiles. | "Everything's backed by reproducible evaluation — 300 episodes per agent, training curves, a twenty-seed final matrix. It's all in the repo." |

## Alternative 30-second cut (for the README hero GIF)

The README GIF should be short and loop cleanly. Use just the
gauntlet payoff:

| Time | Shot |
| --- | --- |
| 0:00–0:03 | Dashboard, click **Gauntlet** quick-pick |
| 0:03–0:06 | Click **Start live** |
| 0:06–0:24 | Ego spawns stuck in lane 0, then escapes and accelerates — the DecisionTicker flashing CHANGE LEFT as it does |
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
