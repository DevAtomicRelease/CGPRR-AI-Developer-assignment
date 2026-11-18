"""Turn captured frames + narration WAVs into an MP4 (H.264 + AAC)."""

import json
import pathlib
import subprocess
import sys

import imageio_ffmpeg

V = pathlib.Path(__file__).parent
OUT = pathlib.Path(sys.argv[1])
ff = imageio_ffmpeg.get_ffmpeg_exe()
d = json.loads((V / "timeline.json").read_text())
frames, timeline, stop = d["frames"], d["timeline"], d["stop"]
v0 = frames[0][0]

lines = []
for i, (ts, name) in enumerate(frames):
    nxt = frames[i + 1][0] if i + 1 < len(frames) else stop
    lines.append(f"file 'frames/{name}'\nduration {max(nxt - ts, 0.001):.4f}")
lines.append(f"file 'frames/{frames[-1][1]}'")  # concat demuxer needs the last file repeated
(V / "frames.txt").write_text("\n".join(lines))

cmd = [ff, "-y", "-f", "concat", "-safe", "0", "-i", str(V / "frames.txt")]
for t in timeline:
    cmd += ["-i", str(V / "audio" / f"{t['id']}.wav")]
parts = []
for i, t in enumerate(timeline, start=1):
    ms = max(0, int((t["t"] - v0) * 1000))
    parts.append(f"[{i}:a]adelay={ms}|{ms},aresample=44100[a{i}]")
mix = "".join(f"[a{i}]" for i in range(1, len(timeline) + 1))
fc = ("[0:v]fps=25,scale=in_range=full:out_range=tv,format=yuv420p[v];" + ";".join(parts)
      + f";{mix}amix=inputs={len(timeline)}:normalize=0,volume=1.3,alimiter=limit=0.89:level=false[aout]")
cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "[aout]", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-t", f"{stop - v0:.2f}", str(OUT)]
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
print("wrote", OUT, f"{stop - v0:.1f}s")
