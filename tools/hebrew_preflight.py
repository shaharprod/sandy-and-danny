#!/usr/bin/env python3
"""
Sandy And Danny - Hebrew pre-flight check (run BEFORE any TTS credits are spent).

Usage:
  python3 hebrew_preflight.py SCRIPT.py [--out report.html] [--asr VOICES_DIR]

SCRIPT.py must define SC = [(n, shot, [(speaker, he, en), ...], sfx), ...]  (the series script format).
Writes:
  <script>_tts_he.json   - per line: text to send to TTS (names/known words pointed, tags kept)
  <script>_subs_he.json  - per line: subtitle text (tags removed, niqqud removed)
  report (HTML)          - every finding, with severity and suggested fix
Exit code 1 if any ERROR was found, so a build script can stop before generating audio.

--asr VOICES_DIR (after TTS): transcribes H01.mp3, H02.mp3 ... with faster-whisper and flags lines
whose transcript differs from the script (word error rate > 0.30, or a character name missing).
"""
import sys, re, json, html, os, argparse, runpy

HERE = os.path.dirname(os.path.abspath(__file__))
LEX = json.load(open(os.path.join(HERE, "he_lexicon.json"), encoding="utf-8"))
NIQ = re.compile(r"[֑-ׇ]")
TAG = re.compile(r"\[[^\]]*\]")
HE_WPS, EN_WPS = 2.3, 2.7


def plain(s):            # no tags, no niqqud
    return re.sub(r"\s+", " ", NIQ.sub("", TAG.sub("", s))).strip()


def words(s):
    return re.findall(r"[א-ת֑-ׇ\"'׳״]+|[A-Za-z]+", s)


def pref(p, tts):
    # prefix letters take sheva; the name's first letter then loses its dagesh (וְדָנִי, לְדוּבִי)
    first = re.match(r"[\u05D0-\u05EA][\u0591-\u05C7]*", tts).group(0)
    return "".join(c + "\u05B0" for c in p) + first.replace("\u05BC", "") + tts[len(first):]


def check_line(k, who, he, en):
    out = []
    p = plain(he); ws = [NIQ.sub("", w) for w in words(TAG.sub("", he))]
    raw_ws = words(TAG.sub("", he))
    # 1. names: bare or with prefix letters
    for w, rw in zip(ws, raw_ws):
        for name, d in LEX["names"].items():
            if w == name and not NIQ.search(rw):
                out.append(("INFO", f"השם \"{name}\" יישלח להקראה מנוקד אוטומטית: {d['tts']} ({d['say']})"))
            elif len(w) > len(name) and w.endswith(name) and all(c in LEX["prefix_letters"] for c in w[:-len(name)]) and not NIQ.search(rw):
                out.append(("ERROR", f"\"{w}\" = שם עם אות שימוש בלי ניקוד. הקריינות עלולה לקרוא אותו לא נכון (כמו לַדובי בפרק 1). לכתוב מנוקד, למשל {pref(w[:-len(name)], d['tts'])}"))
    # 2. risky words
    for w, rw in zip(ws, raw_ws):
        d = LEX["words"].get(w)
        if d and not NIQ.search(rw):
            sev = "ERROR" if d.get("replace") or d.get("tts") is None else "WARN"
            fix = f"להחליף ל\"{d['replace']}\"" if d.get("replace") else f"לנקד: {d['tts']}"
            out.append((sev, f"\"{w}\": {d['why']}. {fix}"))
    # 3. mixing 'you' imperative with a 'we' verb
    imp = [w for w in ws if w in LEX["imperatives_plural"]]
    if imp:
        for i, w in enumerate(ws):
            base = w[1:] if w.startswith("ו") else w
            if base.startswith("נ") and len(base) >= 4 and (i == 0 or ws[i - 1] != "בואו"):
                out.append(("ERROR", f"ערבוב גוף: \"{imp[0]}\" (פנייה אליכם) עם \"{w}\" (אנחנו) באותו משפט. לבחור אחד, למשל \"{imp[0]} ו...ו\" או \"בואו נ...\""))
                break
    # 4. calques from English
    for c in LEX["calques"]:
        if re.search(c["pattern"], p):
            out.append(("WARN", f"נשמע כמו תרגום מאנגלית ({c['why']}). הצעה: {c['suggest']}"))
    # 5. latin letters inside Hebrew
    if re.search(r"[A-Za-z]", p):
        out.append(("ERROR", "יש אותיות באנגלית בתוך המשפט העברי (הקריינות תקרא אותן במבטא אנגלי)"))
    # 6. kid-friendly length and HE/EN timing gap
    nh, ne = len(words(p)), len(words(plain(en)))
    if nh > 12:
        out.append(("WARN", f"משפט ארוך לגיל 4-7 ({nh} מילים). עדיף לפצל לשני משפטים"))
    th, te = nh / HE_WPS, ne / EN_WPS
    if abs(th - te) > 1.5:
        out.append(("INFO", f"פער אורך בין השפות: עברית כ-{th:.1f} ש' מול אנגלית כ-{te:.1f} ש'. לוודא שהסצנה מספיק ארוכה לשתיהן"))
    # 7. partial niqqud -> subtitles must be stripped (the tool does it)
    if NIQ.search(TAG.sub("", he)):
        out.append(("INFO", "יש ניקוד חלקי במשפט. בכתובית הוא יוסר אוטומטית (קובץ subs_he)"))
    return out


def tts_text(he):
    tags = "".join(TAG.findall(he))
    body = TAG.sub("", he).strip()
    def rep(m):
        w = m.group(0)
        if NIQ.search(w): return w
        d = LEX["names"].get(w)
        return d["tts"] if d else w
    body = re.sub(r"[א-ת֑-ׇ]+", rep, body)
    return (tags + " " + body).strip()


def asr_check(lines, vdir):
    from faster_whisper import WhisperModel
    import difflib
    m = WhisperModel("small", device="cpu", compute_type="int8")
    res = []
    for k, (who, he, en) in enumerate(lines, 1):
        f = next((os.path.join(vdir, f"H{k:02d}{e}") for e in (".mp3", ".wav") if os.path.exists(os.path.join(vdir, f"H{k:02d}{e}"))), None)
        if not f:
            res.append((k, "WARN", "אין קובץ קול")); continue
        segs, _ = m.transcribe(f, language="he", beam_size=5)
        got = " ".join(s.text for s in segs)
        a = re.findall(r"[א-ת]+", plain(he)); b = re.findall(r"[א-ת]+", got)
        wer = 1 - difflib.SequenceMatcher(None, a, b).ratio()
        miss = [n for n in LEX["names"] if any(x.endswith(n) for x in a) and not any(n in x for x in b)]
        if wer > 0.30 or miss:
            res.append((k, "ERROR", f"התמלול שונה מהתסריט (שגיאה {wer:.0%}{', חסר השם ' + ', '.join(miss) if miss else ''}). נשמע: \"{got.strip()}\""))
        else:
            res.append((k, "OK", f"תואם ({wer:.0%}): \"{got.strip()}\""))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script"); ap.add_argument("--out"); ap.add_argument("--asr")
    a = ap.parse_args()
    SC = runpy.run_path(a.script)["SC"]
    lines = [(who, he, en) for sc in SC for (who, he, en) in sc[2]]
    base = os.path.splitext(a.script)[0]
    rows, nerr = [], 0
    for k, (who, he, en) in enumerate(lines, 1):
        for sev, msg in check_line(k, who, he, en):
            rows.append((k, who, plain(he), sev, msg)); nerr += sev == "ERROR"
    if a.asr:
        for k, sev, msg in asr_check(lines, a.asr):
            if sev != "OK":
                rows.append((k, lines[k - 1][0], plain(lines[k - 1][1]), sev, "בדיקת שמיעה: " + msg)); nerr += sev == "ERROR"
    json.dump([tts_text(he) for _, he, _ in lines], open(base + "_tts_he.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump([plain(he) for _, he, _ in lines], open(base + "_subs_he.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    nw = sum(r[3] == "WARN" for r in rows)
    print(f"{len(lines)} lines | ERROR {nerr} | WARN {nw} | INFO {sum(r[3]=='INFO' for r in rows)}")
    for r in rows:
        if r[3] != "INFO": print(f"H{r[0]:02d} {r[1]:8} {r[3]:5} {r[2]}  ->  {r[4]}")
    if a.out:
        col = {"ERROR": "#c62828", "WARN": "#b26a00", "INFO": "#4a6fa5"}
        tr = "".join(f"<tr><td>H{k:02d}</td><td>{html.escape(w)}</td><td>{html.escape(t)}</td><td style='color:{col[s]};font-weight:bold'>{s}</td><td>{html.escape(m)}</td></tr>" for k, w, t, s, m in rows)
        open(a.out, "w", encoding="utf-8").write(
            f"<!doctype html><html lang='he' dir='rtl'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>בדיקת עברית · {html.escape(os.path.basename(a.script))}</title><style>body{{font-family:Arial,sans-serif;margin:16px;background:#fff;color:#222}}"
            f"table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ddd;padding:6px;vertical-align:top}}th{{background:#f4ede4}}</style>"
            f"<h1>בדיקת עברית לפני הקלטה</h1><p>{len(lines)} משפטים · שגיאות {nerr} · אזהרות {nw}</p>"
            f"<table><tr><th>#</th><th>דמות</th><th>משפט</th><th>חומרה</th><th>מה לבדוק</th></tr>{tr}</table></html>")
    sys.exit(1 if nerr else 0)


if __name__ == "__main__":
    main()
