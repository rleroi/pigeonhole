"""Compare every model in serve.MODELS on Dutch + English intent examples.

Usage: python bench.py [model ...]     (default: all models)

Each sentence is tested 3x: without punctuation, with "." and with "?", to
measure sensitivity to punctuation. The Dutch test sentences are intentional.
"""
import json
import sys
import time
from collections import defaultdict

from serve import MODELS, ClassifyRequest, get_model, run_backend

LABELS_EN = ["cancel", "refund", "question", "complaint", "praise"]
# Same labels in Dutch, same order
LABELS_NL = ["annuleren", "terugbetaling", "vraag", "klacht", "compliment"]

# (text without punctuation, index of the correct label)
NL = [
    # cancel (polite requests phrased as a question still mean cancel)
    ("Ik wil mijn bestelling annuleren", 0),
    ("Kunt u mijn abonnement opzeggen", 0),
    ("Mag ik mijn afspraak van morgen annuleren", 0),
    ("Zeg mijn lidmaatschap per direct op", 0),
    ("Ik heb besloten om niet door te gaan met de bestelling", 0),
    ("Kan mijn reservering geannuleerd worden", 0),
    ("Stop alsjeblieft mijn abonnement", 0),
    ("Ik wil de koop ongedaan maken voordat het wordt verzonden", 0),
    # refund
    ("Ik wil mijn geld terug", 1),
    ("Waar blijft mijn terugbetaling", 1),
    ("Ik heb het product teruggestuurd, wanneer krijg ik mijn geld", 1),
    ("Graag het aankoopbedrag retour op mijn rekening", 1),
    ("Kan ik een restitutie krijgen voor deze kapotte laptop", 1),
    ("Ik heb twee keer betaald, wilt u een van de bedragen terugstorten", 1),
    ("Mijn creditnota is nog niet binnen", 1),
    ("Ik eis vergoeding van de kosten terug", 1),
    # question
    ("Hoe lang duurt de levering", 2),
    ("Wat zijn jullie openingstijden", 2),
    ("Is dit product ook in het blauw verkrijgbaar", 2),
    ("Kan ik met iDEAL betalen", 2),
    ("Wat kost verzending naar België", 2),
    ("Hoe werkt de garantie", 2),
    ("Hebben jullie een winkel in Utrecht", 2),
    ("Welke maten zijn er beschikbaar", 2),
    # complaint
    ("Mijn pakket is beschadigd aangekomen", 3),
    ("De klantenservice neemt al een week de telefoon niet op", 3),
    ("Dit is de derde keer dat de levering te laat is", 3),
    ("Het product werkt niet zoals beloofd", 3),
    ("Ik ben erg ontevreden over jullie service", 3),
    ("De medewerker was onbeleefd tegen mij", 3),
    ("Ik heb de verkeerde maat ontvangen", 3),
    ("De website crasht steeds bij het afrekenen", 3),
    # praise
    ("Super snelle levering, top gedaan", 4),
    ("Wat een vriendelijke medewerker, bedankt voor de hulp", 4),
    ("Ik ben zeer tevreden met mijn aankoop", 4),
    ("Fantastische service, ik beveel jullie zeker aan", 4),
    ("Het product is precies wat ik zocht, geweldig", 4),
    ("Mooi pakket, netjes verpakt", 4),
    ("Beste webshop waar ik ooit besteld heb", 4),
    ("Dank jullie wel, het probleem is perfect opgelost", 4),
]

EN = [
    ("I want to cancel my order", 0),
    ("Please close my account", 0),
    ("Can you cancel my subscription", 0),
    ("I want my money back", 1),
    ("When will I get my refund", 1),
    ("Can I get a reimbursement for the broken item", 1),
    ("How long does shipping take", 2),
    ("What are your opening hours", 2),
    ("Do you ship to Germany", 2),
    ("My package arrived damaged", 3),
    ("Your support never answers the phone", 3),
    ("The app keeps crashing", 3),
    ("Great service, thank you", 4),
    ("I love this product", 4),
    ("Very friendly and helpful staff", 4),
]

VARIANTS = {"": "", ".": ".", "?": "?"}


def predict(name, model, text, labels):
    res = run_backend(name, model, ClassifyRequest(text=text, labels=labels, model=name))
    return res[0]["label"], res[0]["score"]


def run_model(name):
    model = get_model(name)
    rows = []
    t0 = time.time()
    n = 0
    for lang, data, label_sets in (("nl", NL, {"en": LABELS_EN, "nl": LABELS_NL}), ("en", EN, {"en": LABELS_EN})):
        for label_lang, labels in label_sets.items():
            for text, gold in data:
                for vname, suffix in VARIANTS.items():
                    pred, score = predict(name, model, text + suffix, labels)
                    rows.append({
                        "text": text, "variant": vname, "lang": lang, "label_lang": label_lang,
                        "gold": labels[gold], "pred": pred, "score": score, "ok": pred == labels[gold],
                        "pred_is_question": pred in ("question", "vraag"),
                        "gold_is_question": gold == 2,
                    })
                    n += 1
    return rows, (time.time() - t0) / n


def pct(xs):
    xs = list(xs)
    return 100 * sum(xs) / len(xs) if xs else float("nan")


def summarize(name, rows, sec_per_call):
    def acc(**kw):
        return pct(r["ok"] for r in rows if all(r[k] == v for k, v in kw.items()))

    # punctuation flip: same sentence, different prediction with "?" vs without
    by = defaultdict(dict)
    for r in rows:
        by[(r["text"], r["label_lang"])][r["variant"]] = r["pred"]
    flips = pct(v[""] != v["?"] for v in by.values())
    # non-questions that tip over to 'question' because of the "?"
    nq = [r for r in rows if not r["gold_is_question"]]
    q_plain = pct(r["pred_is_question"] for r in nq if r["variant"] == "")
    q_mark = pct(r["pred_is_question"] for r in nq if r["variant"] == "?")
    return {
        "model": name,
        "nl_en_labels": acc(lang="nl", label_lang="en"),
        "nl_nl_labels": acc(lang="nl", label_lang="nl"),
        "en": acc(lang="en", label_lang="en"),
        "no_punct": acc(variant=""),
        "qmark": acc(variant="?"),
        "flip_pct": flips,
        "question_bias": f"{q_plain:.0f}% -> {q_mark:.0f}%",
        "ms_per_call": 1000 * sec_per_call,
    }


def main():
    names = sys.argv[1:] or list(MODELS)
    summaries, all_rows = [], {}
    for name in names:
        print(f"== {name}", flush=True)
        try:
            rows, spc = run_model(name)
        except Exception as e:  # model failed to load, etc.
            print(f"   FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        all_rows[name] = rows
        s = summarize(name, rows, spc)
        summaries.append(s)
        print("  ", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
        with open("bench_rows.json", "w") as f:
            json.dump(all_rows, f, ensure_ascii=False)

    print("\n| model | NL text, EN labels | NL text, NL labels | EN | no punctuation | with ? | flip by ? | non-questions -> question | ms/call |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in sorted(summaries, key=lambda s: -s["nl_en_labels"]):
        print(f"| {s['model']} | {s['nl_en_labels']:.0f}% | {s['nl_nl_labels']:.0f}% | {s['en']:.0f}% | "
              f"{s['no_punct']:.0f}% | {s['qmark']:.0f}% | {s['flip_pct']:.0f}% | {s['question_bias']} | {s['ms_per_call']:.0f} |")


if __name__ == "__main__":
    main()
