# A quick CA analysis of a section

Correspondence analysis of the research interests declared by the members of a
university section in a 2022 internal survey. `data.csv` holds the anonymised
answers: one row per member (`member_01`, ...), one column per research topic,
values are declared interest levels (1-7). The raw survey export is not part of
this repository.

```
pip install -r requirements.txt
python main.py
```

Produces `section.pdf` (2D map) and `section.html` (interactive 3D map).
