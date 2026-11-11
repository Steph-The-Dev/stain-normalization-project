# Prüfung: Hat Spital 1 dunkelgraue statt schwarze Flächen (angehobener Schwarzwert), die der Filter mit 30 nicht erfasst?
# In Notebook 1 als neue Zelle ausführen, nachdem die Daten geladen sind (wds, meta). Kein Training, rund 1–2 Minuten.
rng = np.random.default_rng(0)

# 1) Wie verteilen sich die dunkelsten Pixel? Anteil der Pixel nach hellstem Kanal, je 300 Ausschnitte pro Spital
rows = []
for h in range(5):
    idx = rng.choice(np.where(meta["hospital"] == h)[0], 300, replace=False)
    px = np.concatenate([tl.load_image(wds, i).reshape(-1, 3) for i in idx])
    mx = px.max(axis=1)
    dunkel = px[mx < 60]
    rows.append({"spital": h,
                 "<30 (%)": 100 * (mx < 30).mean(), "30–45 (%)": 100 * ((mx >= 30) & (mx < 45)).mean(),
                 "45–60 (%)": 100 * ((mx >= 45) & (mx < 60)).mean(),
                 "RGB der Pixel <60": str(dunkel.mean(axis=0).round(0)) if len(dunkel) else "–",
                 "Buntheit (max−min) <60": (dunkel.max(axis=1) - dunkel.min(axis=1)).mean().round(1) if len(dunkel) else None})
display(pd.DataFrame(rows).round(3))

# 2) H-Vektor der Slides aus Spital 1 bei verschiedenen Schwellen (gleiche 32 Ausschnitte pro Slide wie im Notebook)
slides1 = np.unique(meta["slide"][meta["hospital"] == 1])
res = []
for t in [None, 30, 45, 60, 80]:
    params = tl.fit_slide_params(wds, meta, slides=slides1, k=32, verbose=False, dark_thresh=t)
    for s, p in params.items():
        res.append({"schwelle": 0 if t is None else t, "slide": s,
                    "H_winkel": p.angles_deg()[0], "Hmax": p.maxC[0]})
res = pd.DataFrame(res)
print("H-Winkel (°) pro Slide und Schwelle – normal sind 10–16°:")
display(res.pivot(index="slide", columns="schwelle", values="H_winkel").round(1))
print("H-Intensität pro Slide und Schwelle – normal sind 1,5–2,5:")
display(res.pivot(index="slide", columns="schwelle", values="Hmax").round(2))
res.to_csv("results/pruefung_schwarzwert_spital1.csv", index=False)
