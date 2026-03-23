# UI Handoff

## Delivered Scope

- main shell
- category sidebar
- widget catalog
- workflow canvas
- popup widget windows
- active `Data` widgets:
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

## Intentionally Empty Widgets

The following are placeholders by design, not bugs:

- current placeholder widgets in `Transform`, `Visualize`, `Model`, `Evaluate`, `Unsupervised`

They exist to keep:

- the Orange-like information architecture
- stable widget ids
- stable future connection points

## Important Files

- `D:\Projects\portakal\YZM318-Portakal-Data-Mining-Application\src\portakal_app\ui\main_window.py`
- `D:\Projects\portakal\YZM318-Portakal-Data-Mining-Application\src\portakal_app\ui\shell\workflow_canvas.py`
- `D:\Projects\portakal\YZM318-Portakal-Data-Mining-Application\src\portakal_app\ui\shell\workflow_workspace.py`
- `D:\Projects\portakal\YZM318-Portakal-Data-Mining-Application\src\portakal_app\ui\screens\datasets_screen.py`
- `D:\Projects\portakal\YZM318-Portakal-Data-Mining-Application\src\portakal_app\ui\screens\paint_data_screen.py`
- `D:\Projects\portakal\YZM318-Portakal-Data-Mining-Application\src\portakal_app\ui\screens\color_screen.py`
- `D:\Projects\portakal\YZM318-Portakal-Data-Mining-Application\src\portakal_app\ui\screens\file_screen.py`
- `D:\Projects\portakal\YZM318-Portakal-Data-Mining-Application\src\portakal_app\ui\screens\data_table_screen.py`
- `D:\Projects\portakal\YZM318-Portakal-Data-Mining-Application\src\portakal_app\ui\screens\data_info_screen.py`
- `D:\Projects\portakal\YZM318-Portakal-Data-Mining-Application\src\portakal_app\ui\screens\save_data_screen.py`

## Workflow Rules

- connections are manual only
- connection starts from output and ends at compatible input
- input ports accept only one incoming edge
- `File` is a source node and has output only
- `CSV File Import` and `Datasets` are additional source nodes with `Data` output
- port visuals stay small, but hitboxes are intentionally larger for usability
- widgets with input ports only receive dataset content while they have a valid upstream `Data` path
- disconnecting an upstream edge clears the loaded dataset in dependent widgets

## Popup Rules

Popup sizing and centering are intentionally frozen in `workflow_workspace.py`.

- default widgets use one shared popup profile
- `Save Data` uses a smaller profile
- popups are centered against the parent shell
- later teams should not casually change these rules

## Current Data Flow Assumption

- dataset state is introduced by source widgets and then propagated only to connected downstream widgets
- popup footer `Data Preview` is enabled only when a node has a valid upstream `Data` path in the workflow graph
- `Selected Preview` is only exposed by widgets that implement `detailed_data_snapshot()`
- `Paint Data` emits generated datasets back into the same global session state
- `Color` stores visual assignments in dataset annotations
- `Datasets` uses a curated remote catalog with 15 downloadable entries

## Verification

UI baseline is covered by:

- `D:\Projects\portakal\tests\test_main_window.py`

Expected verification command:

```powershell
pytest -q
```
