# YZM318-Portakal-Data-Mining-Application

Bu repository Ankara Üniversitesi 2026 dönemi YZM318 dersi kapsamında kurulmuştur. <br>
Dr. Öğr. Üyesi Bahaeddin Türkoğlu danışmanlığında AÜ YVZM öğrencileri tarafından geliştirilen "Portakal" veri madenciliği uygulamasının versiyonları burada tutulacaktır.<br>
Orijinal açık kaynaklı "Orange Data Mining" uygulamasından esinlenilmiştir (https://orangedatamining.com/). 

<br>
<br>

Created for the YZM318 course (2026 term) at Ankara University, this repository keeps the development versions of the "Portakal" data mining tool. <br>
The project is developed by AU AI&DE students and supervised by Asst. Prof. Bahaeddin Türkoğlu. <br>
The project draws inspiration from the open-source Orange Data Mining platform (https://orangedatamining.com/).

## Development

Orange-inspired desktop shell built with PySide6.

Create a virtual environment and install dependencies:

```powershell
python -m pip install -e .[dev]
```

Run the app:

```powershell
$env:PYTHONPATH='src'
python -m portakal_app
```

Run tests:

```powershell
pytest
```

## Current UI Scope

- The app shell, workflow canvas, popup widget windows, and the current `Data` workflow are implemented.
- Active `Data` widgets:
  - `File`
  - `CSV File Import`
  - `Datasets`
  - `Data Table`
  - `Paint Data`
  - `Data Info`
  - `Rank`
  - `Edit Domain`
  - `Color`
  - `Column Statistics`
  - `Save Data`
- `Datasets` ships with 15 curated downloadable datasets.
- `Paint Data` supports manual point editing with Orange-inspired tools.
- `Color` supports discrete value colors and numeric gradient assignments.
- Remaining widgets in `Transform`, `Visualize`, `Model`, `Evaluate`, and `Unsupervised` are still placeholders.

## Workflow Data Rules

- Data is no longer shown globally in every widget.
- A widget with an input port only receives dataset content while it has a valid upstream `Data` path in the workflow graph.
- If the connection is removed, the widget clears its loaded dataset state.
- Source widgets (`File`, `CSV File Import`, `Datasets`) remain responsible for introducing data into the workflow.

## Frozen Popup Rules

- Popup size and centering behavior are intentionally standardized in `workflow_workspace.py`.
- Default widgets share one popup size profile.
- `Save Data` uses a smaller dedicated popup size profile.
- These sizing rules are considered frozen unless a later requirement explicitly changes popup UX.
