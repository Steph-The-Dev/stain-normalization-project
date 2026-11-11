# Diagnose: Warum fällt Spital 1 mit dem Filter für schwarze Pixel?
# In Notebook 1 als neue Zelle ausführen, nachdem die Zellen bis "Daten laden" gelaufen sind (wds, meta).
# Vergleicht Referenz und Farbparameter pro Slide ohne Filter (stain_params.pkl) und mit Filter (stain_params_dark.pkl)
# und zeigt Ausschnitte aus Spital 1: roh, Macenko alt, Macenko neu.
old = tl.load_pickle("results/stain_params.pkl")
new = tl.load_pickle("results/stain_params_dark.pkl")
fmt = lambda p: f"H {p.angles_deg()[0]:5.1f}°  E {p.angles_deg()[1]:5.1f}°  Hmax {p.maxC[0]:.2f}  Emax {p.maxC[1]:.2f}"
print("Referenz ohne Filter:", fmt(old["ref"]))
print("Referenz mit Filter: ", fmt(new["ref"]))

rows = []
for s in sorted(new["slide_params"]):
    h = int(np.bincount(meta["hospital"][meta["slide"] == s]).argmax())
    po, pn = old["slide_params"].get(s), new["slide_params"].get(s)
    if po is None or pn is None:
        continue
    rows.append({"slide": s, "spital": h,
                 "H_winkel_alt": po.angles_deg()[0], "H_winkel_neu": pn.angles_deg()[0],
                 "E_winkel_alt": po.angles_deg()[1], "E_winkel_neu": pn.angles_deg()[1],
                 "Hmax_alt": po.maxC[0], "Hmax_neu": pn.maxC[0],
                 "Emax_alt": po.maxC[1], "Emax_neu": pn.maxC[1]})
vgl = pd.DataFrame(rows).round(2)
vgl.to_csv("results/nb1_filter_vergleich.csv", index=False)
display(vgl.groupby("spital").mean(numeric_only=True).drop(columns="slide").round(2))

# Anteil sehr dunkler Pixel (hellster Kanal < 30) pro Spital, aus je 200 Ausschnitten
rng = np.random.default_rng(5)
dark = []
for h in range(5):
    idx = rng.choice(np.where(meta["hospital"] == h)[0], 200, replace=False)
    frac = np.mean([(tl.load_image(wds, i).reshape(-1, 3).max(axis=1) < 30).mean() for i in idx])
    dark.append({"spital": h, "anteil_dunkle_pixel_%": round(100 * frac, 3)})
display(pd.DataFrame(dark))

# Bilder: Spital 1, je 6 Ausschnitte – oben roh, Mitte Macenko ohne Filter, unten mit Filter
idx = rng.choice(np.where(meta["hospital"] == 1)[0], 6, replace=False)
fig, ax = plt.subplots(3, 6, figsize=(12, 6.4))
for j, i in enumerate(idx):
    img = tl.load_image(wds, i); s = int(meta["slide"][i])
    ax[0, j].imshow(img)
    ax[1, j].imshow(sl.macenko_apply(img, old["slide_params"][s], old["ref"]))
    ax[2, j].imshow(sl.macenko_apply(img, new["slide_params"][s], new["ref"]))
for a in ax.ravel():
    a.axis("off")
for r, t in enumerate(["roh", "ohne Filter", "mit Filter"]):
    ax[r, 0].text(-20, 48, t, rotation=90, va="center")
plt.suptitle("Spital 1"); plt.tight_layout(); plt.savefig("results/nb1_filter_spital1.png", dpi=150); plt.show()
