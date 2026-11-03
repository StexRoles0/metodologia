"""
Prueba de concepto: DMD y DMDc sobre velocidades Bitcarrier (nov. 2022)
en los tramos del entorno de la intersección de Puente Aranda, Bogotá.

Uso:  python poc_dmdc.py ruta/al/archivo.csv
Salida: carpeta out/ con figuras (PNG) y metricas.json
"""
import sys, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "data_raw/nov2022.parquet"
OUT = "out"
os.makedirs(OUT, exist_ok=True)
DT_MIN = 15
STEPS_DAY = 24 * 60 // DT_MIN  # 96
FESTIVOS = pd.to_datetime(["2022-11-07", "2022-11-14"])  # festivos de nov. 2022 en Colombia

# Tramos del entorno de Puente Aranda (TID: etiqueta corta)
TRAMOS = {
    1001038: "Américas Cr36→Cr62",
    1001039: "Américas Cr62→Cr36",
    1001949: "Américas/Cr50 Cr36→Cl17",
    1000698: "Américas/Cl34 Cr36→Cr28",
    1001715: "Américas/Cl34 Cr28→Cr36",
    1001947: "Cl13/Cr50 Cr62→Cl5A",
    1001948: "Cr50/Cl13 Cl5A→Cr47",
    1001950: "Cr50/Cl13 Cl17→Cr56",
    1000700: "Cl13 Cr56→Cr47",
    1000706: "Cl13 Cr39→Cr47",
    1000707: "Cl13 Cr47→Cr39",
    1000689: "Cr50 Cl17→Cl24",
    1000690: "Cr50 Cl24→Cl17",
    1000701: "Cr50 Cl5A→Cl2C",
    1001730: "Cr50 Cl2C→Cl5A",
    1001968: "NQS Cl3→Cl12",
    1001969: "NQS Cl12→Cl3",
    1000779: "Cl6 Cr36→Cr42",
    1000780: "Cl6 Cr42→Cr36",
    1000691: "Cr36 Américas→Cl19",
    1000692: "Cr36 Cl19→Américas",
}

# ---------------- 1. Carga ----------------
if CSV.endswith(".parquet"):
    df = pd.read_parquet(CSV)
else:
    df = pd.read_csv(CSV, encoding="utf-8-sig", low_memory=False,
                     usecols=["TID", "INICIO", "VEL_PROMEDIO", "NUMDISPOSITIVOS"])
    df["t"] = pd.to_datetime(df.INICIO, format="%m/%d/%Y %I:%M:%S %p")
n_registros_total = len(df)
n_tramos_total = df.TID.nunique()
df = df[df.TID.isin(TRAMOS)]
grid = pd.date_range("2022-11-01 00:00", "2022-11-30 23:45", freq=f"{DT_MIN}min")
V = df.pivot_table(index="t", columns="TID", values="VEL_PROMEDIO", aggfunc="mean").reindex(grid)
V = V[[c for c in TRAMOS if c in V.columns]]
falt_inicial = float(V.isna().mean().mean())

# ---------------- 2. Preprocesamiento ----------------
V = V.clip(lower=1, upper=90)                       # valores físicamente plausibles (km/h)
V = V.interpolate(limit=4, limit_direction="both")  # huecos de hasta 1 h
tod = V.index.hour * 4 + V.index.minute // 15
for c in V.columns:                                 # huecos largos: mediana de la misma franja
    med = V[c].groupby(tod).transform("median")
    V[c] = V[c].fillna(med)
V = V.rolling(2, min_periods=1).mean()               # suavizado leve (30 min) por ruido de pocos dispositivos
assert not V.isna().any().any()

# velocidad de flujo libre: percentil 85 entre 00:00 y 04:59
night = V.index.hour < 5
vff = V[night].quantile(0.85)
CI = V / vff  # índice v/v_ff

N, m = V.shape[1], V.shape[0]
labels = [TRAMOS[c] for c in V.columns]

# calendario
day = V.index.normalize()
no_lab = ((V.index.dayofweek >= 5) | day.isin(FESTIVOS)).astype(float)
hrs = (V.index.hour + V.index.minute / 60).values
U = np.vstack([
    np.sin(2 * np.pi * hrs / 24), np.cos(2 * np.pi * hrs / 24),
    np.sin(4 * np.pi * hrs / 24), np.cos(4 * np.pi * hrs / 24),
    np.sin(6 * np.pi * hrs / 24), np.cos(6 * np.pi * hrs / 24),
    no_lab,
    no_lab * np.sin(2 * np.pi * hrs / 24), no_lab * np.cos(2 * np.pi * hrs / 24),
])  # (q, m)
q = U.shape[0]

# ---------------- 3. Matriz de datos ----------------
X = V.values.T  # filas: tramos (N), columnas: instantes de 15 min (m)
split = 21 * STEPS_DAY  # entrenamiento: 1–21 nov; prueba: 22–30 nov
mu = X[:, :split].mean(axis=1, keepdims=True)
Xc = X - mu

# SVD (datos centrados de entrenamiento)
Uu, S, Vt = np.linalg.svd(Xc[:, :split], full_matrices=False)
energia = np.cumsum(S**2) / np.sum(S**2)
r90 = int(np.searchsorted(energia, 0.90) + 1)
err_rec = {int(r): float(np.linalg.norm(Xc[:, :split] - (Uu[:, :r] * S[:r]) @ Vt[:r]) /
                          np.linalg.norm(Xc[:, :split])) for r in [1, 2, 3, 5, 8]}

# ---------------- 4. Modelos ----------------
def fit_dmd(Xtr, r):
    X1, X2 = Xtr[:, :-1], Xtr[:, 1:]
    u, s, vt = np.linalg.svd(X1, full_matrices=False)
    u, s, vt = u[:, :r], s[:r], vt[:r]
    A = X2 @ vt.T @ np.diag(1 / s) @ u.T
    return A

def fit_dmdc(Xtr, Utr, p):
    X1, X2 = Xtr[:, :-1], Xtr[:, 1:]
    Om = np.vstack([X1, Utr[:, :-1]])
    u, s, vt = np.linalg.svd(Om, full_matrices=False)
    u, s, vt = u[:, :p], s[:p], vt[:p]
    G = X2 @ vt.T @ np.diag(1 / s) @ u.T
    n = Xtr.shape[0]
    return G[:, :n], G[:, n:]

def forecast(A, B, x0, Ufut, h):
    preds, x = [], x0.copy()
    for k in range(h):
        x = A @ x + (B @ Ufut[:, k] if B is not None else 0)
        preds.append(x.copy())
    return np.array(preds)  # (h, N)

H = 16  # horizonte: 16 pasos = 4 h
origins = np.arange(split, m - H, 4)  # un origen cada hora en el periodo de prueba
prof_tr = pd.DataFrame(Xc[:, :split].T).groupby(
    [no_lab[:split].astype(int), tod[:split]]).mean()

def evaluate(model):
    err = np.zeros((len(origins), H, N))
    for i, o in enumerate(origins):
        true = Xc[:, o + 1:o + 1 + H].T
        if model == "persistencia":
            pred = np.repeat(Xc[:, o][None], H, 0)
        elif model == "perfil":
            idx = [(int(no_lab[o + 1 + k]), int(tod[o + 1 + k])) for k in range(H)]
            pred = prof_tr.loc[idx].values
        elif model == "dmd":
            pred = forecast(A_dmd, None, Xc[:, o], None, H)
        else:
            pred = forecast(A_c, B_c, Xc[:, o], U[:, o:o + H], H)
        err[i] = pred - true
    rmse_h = np.sqrt((err**2).mean(axis=(0, 2)))           # por horizonte
    rmse_o = np.sqrt((err**2).mean(axis=(1, 2)))           # por origen (dispersión)
    mae = np.abs(err).mean()
    return rmse_h, rmse_o, mae, err

# selección de rangos con validación (días 19–21 del entrenamiento)
val_split = 18 * STEPS_DAY
def val_score(kind, r):
    Xtr = Xc[:, :val_split]
    if kind == "dmd":
        A = fit_dmd(Xtr, r); B = None
    else:
        A, B = fit_dmd_c = fit_dmdc(Xtr, U[:, :val_split], r)
    e = []
    for o in range(val_split, split - H, 4):
        p = forecast(A, B, Xc[:, o], U[:, o:o + H] if B is not None else None, H)
        e.append(((p - Xc[:, o + 1:o + 1 + H].T) ** 2).mean())
    return float(np.sqrt(np.mean(e)))

sens_dmd = {r: val_score("dmd", r) for r in range(2, N + 1, 2)}
sens_dmdc = {p: val_score("dmdc", p) for p in range(4, N + q + 1, 2)}
r_best = min(sens_dmd, key=sens_dmd.get)
p_best = min(sens_dmdc, key=sens_dmdc.get)

A_dmd = fit_dmd(Xc[:, :split], r_best)
A_c, B_c = fit_dmdc(Xc[:, :split], U[:, :split], p_best)

res = {k: evaluate(k) for k in ["persistencia", "perfil", "dmd", "dmdc"]}

# ---------------- 5. Contrafactual (mecanismo DMDc) ----------------
# Domingo 27 nov (prueba): simular con la entrada "no laborable" = 0
d0 = pd.Timestamp("2022-11-27")
i0 = np.where(V.index == d0)[0][0]
Ucf = U[:, i0:i0 + STEPS_DAY].copy(); Ucf[6:] = 0
def free_run(A, B, x0, Useq):
    xs, x = [], x0.copy()
    for k in range(Useq.shape[1]):
        x = A @ x + B @ Useq[:, k]; xs.append(x.copy())
    return np.array(xs)
sim_obs = free_run(A_c, B_c, Xc[:, i0 - 1], U[:, i0:i0 + STEPS_DAY]) + mu.T
sim_cf = free_run(A_c, B_c, Xc[:, i0 - 1], Ucf) + mu.T
obs_dom = X[:, i0:i0 + STEPS_DAY].T
# referencia: promedio de días laborables de la semana de prueba (22–25 nov)
lab_days = [pd.Timestamp(f"2022-11-{d}") for d in (22, 23, 24, 25)]
lab_prof = np.mean([X[:, np.where(V.index == d)[0][0]:np.where(V.index == d)[0][0] + STEPS_DAY].T
                    for d in lab_days], axis=0)

# ---------------- 6. Métricas ----------------
eig_dmd = np.linalg.eigvals(A_dmd); eig_c = np.linalg.eigvals(A_c)
met = {
    "datos": {"registros_mes_toda_la_red": int(n_registros_total), "tramos_red": int(n_tramos_total),
              "tramos_usados": int(N), "instantes": int(m), "resolucion_min": DT_MIN,
              "faltantes_iniciales_pct": round(100 * falt_inicial, 2),
              "entrenamiento": "1–21 nov 2022 (2016 instantes)", "prueba": "22–30 nov 2022 (864 instantes)"},
    "svd": {"r_90pct": r90, "energia_r": {int(r): round(float(energia[r - 1]), 4) for r in [1, 2, 3, 5, 8]},
            "error_reconstruccion_rel": {k: round(v, 4) for k, v in err_rec.items()}},
    "rangos": {"r_dmd": int(r_best), "p_dmdc": int(p_best),
               "sens_dmd": {int(k): round(v, 3) for k, v in sens_dmd.items()},
               "sens_dmdc": {int(k): round(v, 3) for k, v in sens_dmdc.items()}},
    "rmse_por_horizonte_kmh": {k: [round(float(x), 3) for x in v[0]] for k, v in res.items()},
    "rmse_medio_kmh": {k: round(float(np.sqrt((v[3]**2).mean())), 3) for k, v in res.items()},
    "rmse_dispersion_std_origenes": {k: round(float(v[1].std()), 3) for k, v in res.items()},
    "mae_kmh": {k: round(float(v[2]), 3) for k, v in res.items()},
    "n_origenes_prueba": int(len(origins)),
    "eig_dmd_abs_max": round(float(np.abs(eig_dmd).max()), 4),
    "eig_dmdc_abs_max": round(float(np.abs(eig_c).max()), 4),
    "vff_kmh": {TRAMOS[c]: round(float(vff[c]), 1) for c in V.columns},
    "ci_medio_laborable_pico_tarde": None,
}
pico = (V.index.hour >= 17) & (V.index.hour < 20) & (no_lab == 0)
met["ci_medio_laborable_pico_tarde"] = {TRAMOS[c]: round(float(CI[c][pico].mean()), 2) for c in V.columns}
# efecto contrafactual en el domingo
met["contrafactual_domingo"] = {
    "vel_media_obs_kmh": round(float(obs_dom.mean()), 2),
    "vel_media_sim_obs_kmh": round(float(sim_obs.mean()), 2),
    "vel_media_sim_laborable_kmh": round(float(sim_cf.mean()), 2),
    "vel_media_laborables_reales_kmh": round(float(lab_prof.mean()), 2),
    "rmse_sim_cf_vs_laborables_reales": round(float(np.sqrt(((sim_cf - lab_prof)**2).mean())), 3),
    "rmse_dom_obs_vs_laborables_reales": round(float(np.sqrt(((obs_dom - lab_prof)**2).mean())), 3),
}
json.dump(met, open(f"{OUT}/metricas.json", "w"), ensure_ascii=False, indent=1)

# ---------------- 7. Figuras ----------------
plt.rcParams.update({"font.size": 17, "axes.titlesize": 18, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#FBF8F2", "axes.facecolor": "#FBF8F2", "savefig.facecolor": "#FBF8F2", "lines.linewidth": 2})
C = {"persistencia": "#9a9a9a", "perfil": "#d08a2c", "dmd": "#4a7fb5", "dmdc": "#c2185b"}
NAMES = {"persistencia": "Persistencia", "perfil": "Perfil histórico", "dmd": "DMD", "dmdc": "DMDc"}

# F1 espectro SVD
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(np.arange(1, len(S) + 1), S, "o-", color="#333")
ax[0].set_xlabel("Índice i"); ax[0].set_ylabel("Valor singular σᵢ"); ax[0].set_title("Espectro de valores singulares")
ax[1].plot(np.arange(1, len(S) + 1), 100 * energia, "o-", color="#c2185b")
ax[1].axhline(90, ls="--", color="#999"); ax[1].axvline(r90, ls=":", color="#999")
ax[1].set_xlabel("Rango r"); ax[1].set_ylabel("Energía acumulada (%)"); ax[1].set_title(f"90 % de energía con r = {r90}")
fig.tight_layout(); fig.savefig(f"{OUT}/f1_svd.png", dpi=200); plt.close(fig)

# F2 RMSE vs horizonte
fig, ax = plt.subplots(figsize=(8, 4.6))
hh = np.arange(1, H + 1) * DT_MIN
for k in res:
    rmse_o_h = np.sqrt((res[k][3]**2).mean(axis=2))  # (origenes,H)
    ax.plot(hh, res[k][0], "o-", ms=4, color=C[k], label=NAMES[k], lw=2.2 if k == "dmdc" else 1.6)
    ax.fill_between(hh, np.percentile(rmse_o_h, 25, 0), np.percentile(rmse_o_h, 75, 0), color=C[k], alpha=0.10)
ax.set_xlabel("Horizonte de predicción (min)"); ax.set_ylabel("RMSE (km/h)")
ax.set_title(f"Error de predicción en prueba (22–30 nov, {len(origins)} orígenes, {N} tramos)")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{OUT}/f2_rmse_horizonte.png", dpi=200); plt.close(fig)

# F3 eigenvalores
fig, ax = plt.subplots(figsize=(5, 5))
th = np.linspace(0, 2 * np.pi, 400); ax.plot(np.cos(th), np.sin(th), color="#bbb")
ax.scatter(eig_dmd.real, eig_dmd.imag, color=C["dmd"], label="DMD", s=36)
ax.scatter(eig_c.real, eig_c.imag, color=C["dmdc"], marker="x", label="DMDc (A)", s=46)
ax.set_aspect("equal"); ax.set_xlabel("Re(λ)"); ax.set_ylabel("Im(λ)")
ax.set_title("Eigenvalores frente al círculo unitario"); ax.legend(frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/f3_eigen.png", dpi=200); plt.close(fig)

# F4 predicción vs real (un tramo, un día de prueba)
j = labels.index("Américas Cr36→Cr62")
d1 = pd.Timestamp("2022-11-23 05:00"); o = np.where(V.index == d1)[0][0]
h_long = 16
fig, ax = plt.subplots(figsize=(10, 4.2))
tt = V.index[o - 8:o + 15 * 4]
ax.plot(tt, X[j, o - 8:o + 15 * 4], color="#222", lw=2, label="Observado")
for k_o in range(o, o + 14 * 4, h_long):
    p = forecast(A_c, B_c, Xc[:, k_o], U[:, k_o:k_o + h_long], h_long)[:, j] + mu[j, 0]
    ax.plot(V.index[k_o + 1:k_o + 1 + h_long], p, color=C["dmdc"], lw=2,
            label="DMDc (4 h adelante)" if k_o == o else None)
    pp = np.repeat(X[j, k_o], h_long)
    ax.plot(V.index[k_o + 1:k_o + 1 + h_long], pp, color=C["persistencia"], lw=1.4, ls="--",
            label="Persistencia" if k_o == o else None)
ax.set_ylabel("Velocidad (km/h)"); ax.set_xlabel("Hora (miércoles 23 nov 2022)")
ax.set_title("Av. Américas, Cr 36 → Cr 62 (cruza Puente Aranda)")
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")); ax.legend(frameon=False, ncol=3)
fig.tight_layout(); fig.savefig(f"{OUT}/f4_pred_tramo.png", dpi=200); plt.close(fig)

# F5 contrafactual domingo
fig, ax = plt.subplots(figsize=(10, 4.2))
hx = np.arange(STEPS_DAY) * DT_MIN / 60
ax.plot(hx, obs_dom[:, j], color="#222", lw=2, label="Domingo 27 nov observado")
ax.plot(hx, sim_cf[:, j], color=C["dmdc"], lw=2.2, label="DMDc con u(no laborable) = 0")
ax.plot(hx, lab_prof[:, j], color="#4a7fb5", lw=1.6, ls="--", label="Promedio laborables reales (22–25 nov)")
ax.set_xlabel("Hora del día"); ax.set_ylabel("Velocidad (km/h)"); ax.set_xticks(range(0, 25, 3))
ax.set_title("Contrafactual: ¿cómo habría sido el domingo como día laborable?")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{OUT}/f5_contrafactual.png", dpi=200); plt.close(fig)

# F6 mapa de calor índice de congestión (perfil laborable)
prof = CI[no_lab == 0].groupby(tod[no_lab == 0]).mean()
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(prof.T.values, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=1.0,
               extent=[0, 24, len(labels) - 0.5, -0.5])
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=14)
ax.set_xlabel("Hora del día"); ax.set_xticks(range(0, 25, 3))
ax.set_title("Índice de congestión v/v_ff — día laborable promedio (nov 2022)")
cb = fig.colorbar(im, ax=ax); cb.set_label("v / v_ff (1 = flujo libre)")
fig.tight_layout(); fig.savefig(f"{OUT}/f6_ci_heatmap.png", dpi=200); plt.close(fig)

# F7 modos espaciales (vectores singulares)
fig, ax = plt.subplots(figsize=(9, 8))
for i, col in zip(range(2), ["#c2185b", "#4a7fb5"]):
    ax.barh(np.arange(N) + (0.2 if i else -0.2), Uu[:, i] * np.sign(Uu[:, i].sum()), height=0.4,
            color=col, label=f"u{i+1} ({100*(S[i]**2/np.sum(S**2)):.0f} % energía)")
ax.set_yticks(range(N)); ax.set_yticklabels(labels, fontsize=14); ax.invert_yaxis()
ax.axvline(0, color="#666", lw=0.8); ax.set_xlabel("Peso en el vector singular")
ax.set_title("Patrones espaciales dominantes (SVD)"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/f7_modos.png", dpi=200); plt.close(fig)

# F8 temporal coef de los dos primeros modos
fig, ax = plt.subplots(figsize=(11, 3.6))
tt = V.index[:split]
for i, col in zip(range(2), ["#c2185b", "#4a7fb5"]):
    ax.plot(tt, S[i] * Vt[i] * np.sign(Uu[:, i].sum()), color=col, lw=1, label=f"σ{i+1}·v{i+1}")
ax.set_ylabel("Amplitud"); ax.set_title("Evolución temporal de los dos primeros modos (1–21 nov)")
ax.legend(frameon=False, ncol=2); fig.tight_layout(); fig.savefig(f"{OUT}/f8_modos_t.png", dpi=200); plt.close(fig)

print(json.dumps({k: met[k] for k in ["datos", "svd", "rangos", "rmse_medio_kmh", "mae_kmh",
                                     "rmse_dispersion_std_origenes", "eig_dmd_abs_max", "eig_dmdc_abs_max",
                                     "contrafactual_domingo"]}, ensure_ascii=False, indent=1))
print("RMSE h=1,4,8,16:", {k: [met["rmse_por_horizonte_kmh"][k][i] for i in (0, 3, 7, 15)] for k in res})
