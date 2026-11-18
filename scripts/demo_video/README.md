# How the demo video was made

The demo video records the real application running on the development laptop. Nothing in
the app panels is mocked: every answer, refusal and predictor table on screen is what the app
returned live during the recording. The slides between the app segments (model status, the
CSV line, evaluation output) show terminal output captured on the same laptop just before
recording.

Pipeline (Windows, needs Edge, Ollama and the app running on 127.0.0.1:8080):

1. Capture terminal output into `cap/`: `ollama ps`, an `nvidia-smi` query and
   `python eval/run_eval.py` (saved as `ollama_ps.txt`, `nvidia.txt`, `eval_dev.txt`).
2. `powershell -File tts.ps1 -Dir .` renders each line of `narration.json` to a WAV with the
   offline Windows voice "Microsoft David Desktop" (no cloud text-to-speech).
3. `python make_pages.py <repo>` builds the slide pages from the captured output.
4. `python record.py` (needs `pip install playwright`) drives headless Edge through the app,
   shows each caption, and grabs frames over the DevTools screencast while logging when each
   narration line starts.
5. `python mux.py <out.mp4>` (needs `pip install imageio-ffmpeg`) joins frames and narration
   into an H.264/AAC MP4.

The narration is a synthetic voice. Captions show the same text on screen.
