import json

diar = json.load(open("/data/Famous_Figures/media_analysis/www.youtube.com_watch_v=6qguL6WZ7hY/diarization.json"))
asr = json.load(open("/data/Famous_Figures/media_analysis/www.youtube.com_watch_v=6qguL6WZ7hY/asr_words.json"))

words = asr["words"]

def count_words_in_turn(s, e):
    return sum(1 for w in words if w["start"] >= s and w["end"] <= e)

for t in diar["turns"]:
    c = count_words_in_turn(t["start"], t["end"])
    print(t["speaker"], f"{t['start']:.2f}-{t['end']:.2f}", "words:", c)