"""Optional local summary: turn a scene breakdown into plain English with a small local LLM.

Over MCP the connected LLM (Claude or any other) writes the summary itself, so this is
only for the CLI, when no LLM is connected. Default model: Qwen2.5-1.5B-Instruct
(Apache-2.0, ~3 GB, downloaded on first use).
"""
import torch

DEFAULT_LLM = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPT = """Below is a scene-by-scene breakdown of a video, made by a small vision model.
Each scene lists short captions of frames from that scene. Captions are approximate and
there is no audio.

Write a summary of the video:
1. One or two sentences on what the video is about overall.
2. Then a short line per scene (or per group of similar scenes) with its time range,
   saying what happens.
Only describe what the captions support; say "appears to" when unsure. Don't invent
names, dialogue or sounds.

{breakdown}"""


def summarize(breakdown, model_id=DEFAULT_LLM, device=None, max_new_tokens=400):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16 if device != "cpu" else torch.float32)
    model.to(device).eval()
    msgs = [{"role": "user", "content": PROMPT.format(breakdown=breakdown)}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
