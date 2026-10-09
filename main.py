"""Correspondence analysis of the research interests declared by the members of a
university section (anonymised survey, 2022).

Rows are section members (pseudonymised as member_01, ...), columns are research
topics; a cell is the interest level (1-7) declared for that topic.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# --- DATA
data = pd.read_csv("data.csv")
row_names = list(data["nom"])
col_names = list(data.columns[1:])
data_mat = data.iloc[:, 1:].to_numpy(dtype=float)

# --- CA
n_row, n_col = data_mat.shape
min_row_col = min(n_row, n_col)

# Independence matrix and quotients
n = data_mat.sum()
f = data_mat.sum(axis=1); f = f / f.sum()
fs = data_mat.sum(axis=0); fs = fs / fs.sum()
quot_mat = data_mat / (np.outer(f, fs) * n)

# Transformation of quotients (beta = 1 is the standard CA)
beta = 1
quot_trans = 1 / beta * (quot_mat ** beta - 1)

# Scalar products matrix, eigen decomposition
B = (quot_trans * fs) @ quot_trans.T
K = np.outer(np.sqrt(f), np.sqrt(f)) * B
val_p, vec_p = np.linalg.eigh(K)
idx = val_p.argsort()[::-1]
val_p = np.abs(val_p[idx])[:min_row_col]
vec_p = vec_p[:, idx][:, :min_row_col]

# Row and column coordinates
coord_row = np.outer(1 / np.sqrt(f), np.sqrt(val_p)) * vec_p
coord_col = (quot_trans.T * f) @ coord_row / np.sqrt(val_p)

# Re-centering (useful if beta != 1)
coord_row = coord_row - np.sum(coord_row.T * f, axis=1)
coord_col = coord_col - np.sum(coord_col.T * fs, axis=1)

# --- PLOT 2D
fig, ax = plt.subplots()
ax.scatter(coord_row[:, 0], coord_row[:, 1], alpha=0.5, s=0.1, color="red")
ax.set_xlabel(f"{round(val_p[0] / sum(val_p) * 100, 2)} %")
ax.set_ylabel(f"{round(val_p[1] / sum(val_p) * 100, 2)} %")
for i, txt in enumerate(row_names):
    ax.annotate(txt, (coord_row[i, 0], coord_row[i, 1]), size=2, alpha=1, color="red")
for i, txt in enumerate(col_names):
    ax.annotate(txt, (coord_col[i, 0], coord_col[i, 1]), size=2, alpha=0.5, color="blue")
fig.savefig("section.pdf", dpi=600)

# --- PLOT 3D
annotation_list = [dict(x=coord_col[i, 0], y=coord_col[i, 1], z=coord_col[i, 2], text=txt,
                        textangle=0, ax=0, ay=0, font=dict(color="red", size=12),
                        arrowsize=0.3, arrowwidth=0.3, arrowhead=0)
                   for i, txt in enumerate(col_names)]
fig2 = px.scatter_3d(x=coord_row[:, 0], y=coord_row[:, 1], z=coord_row[:, 2], text=row_names)
fig2.update_layout(scene=dict(annotations=annotation_list))
fig2.write_html("section.html")
