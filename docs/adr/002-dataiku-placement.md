## Decision
Python and SQL own all core data engineering, cleaning, and feature
engineering. Dataiku sits alongside, used only for profiling and
inspecting data structure/quality — not for any step the pipeline
depends on.

## Reasoning
If Dataiku owned a step the pipeline needed, anyone cloning the repo
without Dataiku installed couldn't run it, and visual recipes can't be
unit tested the way Python/SQL can. Keeping Dataiku purely for profiling
and visibility gets the learning value without creating a dependency
the core system relies on.