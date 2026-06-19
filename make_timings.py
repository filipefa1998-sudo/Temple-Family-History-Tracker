#!/usr/bin/env python3
"""
make_timings.py — generate timings.json for the Endowed from on High app.

WHAT IT DOES
  Transcribes each lesson MP3 with word-level timestamps (faster-whisper),
  then aligns those words to the lesson reading text so the app can highlight
  each sentence in exact sync with the narration.

ONE-TIME SETUP (on your own computer, not in a sandbox):
  python3 -m pip install faster-whisper
  # ffmpeg must be installed too:
  #   macOS:  brew install ffmpeg
  #   Windows: download from ffmpeg.org and add to PATH

RUN (from a folder that contains this script, lessons_member.json, and the 7 mp3s):
  python3 make_timings.py
  -> writes timings.json next to index.html. Upload it to the repo. Done.

NOTES
  * First run downloads the speech model (~150 MB for "base"). Reuses it after.
  * "base" is a good balance; for cleaner alignment use model_size="small".
  * The script never needs the internet again after the model is cached.
"""

import json, re, sys, os

MODEL_SIZE = "base"        # try "small" for higher accuracy (slower)
DATA = "lessons_member.json"

# Audio filenames must match what the app expects (the "audio" field per lesson).
# They are read straight from lessons_member.json, so keep the mp3 names unchanged.

# ---- sentence splitter: MUST match the app's splitSentences() exactly ----
ABBR = {"Mr","Mrs","Ms","Dr","St","Sr","Jr","vs","etc","No","Rev","Hon","Gen","Pres","Sis"}
def split_sents(text):
    parts=[]; last=0
    for m in re.finditer(r'([.?!]["”’)]?)(\s+)(?=[A-Z“"‘(])', text):
        i=m.start()
        if text[i-1:i].isupper() and (i-2<0 or text[i-2]==' '): continue
        word=re.split(r'[\s(]', text[max(0,i-5):i])[-1]
        if word in ABBR: continue
        parts.append(text[last:i+len(m.group(1))]); last=m.end()
    parts.append(text[last:])
    return [p.strip() for p in parts if p.strip()]

def lesson_units(L):
    """Reproduce the exact order of .ru elements the app builds."""
    units=[]
    for s in L["sections"]:
        for c in s["callouts"]:
            units.append(c["text"])
        for b in s["blocks"]:
            if b["type"]=="sub":   # h5, not a highlight unit
                continue
            if b["type"]=="q":
                units.append(b["text"])
            else:
                units.extend(split_sents(b["text"]))
    return units

def norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())

def align(units, words):
    """Forward, skip-tolerant alignment of unit start-words to spoken words.
    `words` is a list of (start_seconds, token). Returns one start time per unit."""
    wn = [(t, norm(tok)) for t,tok in words]
    n = len(wn); cur = 0; out=[]
    last_t = 0.0
    for u in units:
        toks = [norm(x) for x in u.split() if norm(x)]
        toks = [t for t in toks if t][:5]   # match on first up-to-5 content words
        start = None
        if toks:
            # search forward from cur for the best window match
            best_i, best_score = -1, 0
            window = min(n, cur + 400)
            for i in range(cur, window):
                if not wn[i][1]: continue
                # score consecutive matches of the unit's leading tokens
                score=0; j=i; k=0
                while k < len(toks) and j < n and (j - i) < len(toks)+6:
                    if wn[j][1] == toks[k] or wn[j][1].startswith(toks[k]) or toks[k].startswith(wn[j][1]):
                        score+=1; k+=1; j+=1
                    else:
                        j+=1
                if score > best_score:
                    best_score, best_i = score, i
                    if score == len(toks): break
            if best_i >= 0 and best_score >= max(1, min(2, len(toks))):
                start = wn[best_i][0]; cur = best_i + 1
        if start is None:
            start = last_t          # carry forward if unmatched (rare)
        start = max(start, last_t)  # enforce monotonic increase
        out.append(round(start, 2))
        last_t = start
    return out

def main():
    if not os.path.exists(DATA):
        sys.exit(f"Missing {DATA} — put it next to this script (download it from the repo).")
    lessons = json.load(open(DATA, encoding="utf-8"))
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper not installed.  Run:  python3 -m pip install faster-whisper")

    print(f"Loading model '{MODEL_SIZE}' (first run downloads it)…")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    timings = {}
    for L in lessons:
        mp3 = L["audio"]
        if not os.path.exists(mp3):
            print(f"  ! Lesson {L['num']}: audio not found ({mp3}) — skipping")
            continue
        print(f"  Lesson {L['num']}: transcribing {mp3} …")
        segments, _ = model.transcribe(mp3, word_timestamps=True, vad_filter=True)
        words=[]
        for seg in segments:
            for w in (seg.words or []):
                words.append((w.start, w.word.strip()))
        units = lesson_units(L)
        starts = align(units, words)
        timings[str(L["id"])] = starts
        print(f"    {len(units)} units aligned to {len(words)} spoken words "
              f"(ends ~{starts[-1] if starts else 0:.0f}s)")

    json.dump(timings, open("timings.json","w"), separators=(",",":"))
    print(f"\nWrote timings.json with {len(timings)} lesson(s). Upload it next to index.html.")

if __name__ == "__main__":
    main()
