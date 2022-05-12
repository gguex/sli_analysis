import pandas as pd
import numpy as np
from itertools import compress
import matplotlib.pyplot as plt
import plotly.express as px


# --- DATA PREPROCESSING

# Loading data
data = pd.read_csv("data/survey_753779_R_data_file.csv")

# Extracting row names
raw_email_list = list(data["EMAIL"])
row_names = [email.split("@")[0] for email in raw_email_list]

# Extracting columns names
data_for_col_names = pd.read_csv("data/results-survey753779.csv")
raw_col_names_list = list(data_for_col_names.columns)[5:]
col_names = [name.split("[")[1][:-1] for name in raw_col_names_list]

# Creating matrix of data
data_mat = np.array(data.iloc[:, 12:])
ind_to_keep = ~np.isnan(data_mat.sum(axis=1))
ind_to_keep[4] = False
data_mat_rem = data_mat[ind_to_keep, :]
row_names_rem = list(compress(row_names, ind_to_keep))
row_names_rem = ["_".join(row_name.split(".")) for row_name in row_names_rem]

# Save the section data
name_df = pd.DataFrame({"nom": row_names_rem})
sup_df = pd.DataFrame(data_mat_rem, columns=col_names).astype(int)
output_df = pd.concat([name_df, sup_df], axis=1)
output_df.to_csv("data.csv", index=False)

# --- CA

# Number of row and col
n_row, n_col = data_mat_rem.shape
min_row_col = min(n_row, n_col)

# Constructing independence matrix
n = np.sum(data_mat_rem)
f = data_mat_rem.sum(axis=1)
f = f / sum(f)
fs = data_mat_rem.sum(axis=0)
fs = fs / sum(fs)
independence_mat = np.outer(f, fs) * n
quot_mat = data_mat_rem / independence_mat

# Transformation of quotients (beta = 1 is the standard CA)
beta = 1
quot_trans = 1 / beta * (quot_mat ** beta - 1)

# Scalar products matrix, eigen decomposition
B = (quot_trans * fs) @ quot_trans.T
K = np.outer(np.sqrt(f), np.sqrt(f)) * B
val_p, vec_p = np.linalg.eig(K)
idx = val_p.argsort()[::-1]
val_p = np.abs(val_p[idx])[:min_row_col]
vec_p = vec_p[:, idx][:, :min_row_col]

# Row coordinates
coord_row = np.outer(1 / np.sqrt(f), np.sqrt(val_p)) * vec_p
# Col coordinates
coord_col = (quot_trans.T * f) @ coord_row / np.sqrt(val_p)

# Re-centering coordinates (useful if beta != 1)
zero_row = np.sum(coord_row.T * f, axis=1)
zero_col = np.sum(coord_col.T * fs, axis=1)
coord_row = coord_row - np.outer(np.ones(coord_row.shape[0]), zero_row)
coord_col = coord_col - np.outer(np.ones(coord_col.shape[0]), zero_col)


# --- PLOT 2D

# Plotting
fig, ax = plt.subplots()
ax.scatter(coord_row[:, 0], coord_row[:, 1], alpha=0.5, s=0.1, color="red")
ax.set_xlabel(f"{round(val_p[0] / sum(val_p) * 100, 2)} %")
ax.set_ylabel(f"{round(val_p[1] / sum(val_p) * 100, 2)} %")

for i, txt in enumerate(row_names_rem):
    ax.annotate(txt, (coord_row[i, 0], coord_row[i, 1]), size=2, alpha=1, color="red")

for i, txt in enumerate(col_names):
    ax.annotate(txt, (coord_col[i, 0], coord_col[i, 1]), size=2, alpha=0.5, color="blue")

fig.savefig('section.pdf', dpi=600)


# --- PLOT 3D

# Make dict for names
annotation_list = [dict(x=coord_col[i, 0], y=coord_col[i, 1], z=coord_col[i, 2], text=txt, textangle=0,
                        ax=0,
                        ay=0,
                        font=dict(color="red", size=12),
                        arrowsize=0.3,
                        arrowwidth=0.3,
                        arrowhead=0) for i, txt in enumerate(col_names)]

# Plotting 3d
fig2 = px.scatter_3d(x=coord_row[:, 0], y=coord_row[:, 1], z=coord_row[:, 2], text=row_names_rem)
fig2.update_layout(scene=dict(annotations=annotation_list))
fig2.write_html("section.html")
