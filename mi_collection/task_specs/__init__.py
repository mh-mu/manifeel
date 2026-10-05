"""``--task-spec`` registry.

An explicit mapping, not a decorator registry, so importing a spec module has no
side effects and the set of supported tasks is readable in one place.
"""

from typing import Dict

from mi_collection.task_spec import TaskSpec
from mi_collection.task_specs.object_search import ObjectSearchSpec
from mi_collection.task_specs.peg_insertion import PegInsertionSpec

TASK_SPECS: Dict[str, type] = {
    PegInsertionSpec.name: PegInsertionSpec,
    ObjectSearchSpec.name: ObjectSearchSpec,
}


def get_task_spec(name: str) -> TaskSpec:
    if name not in TASK_SPECS:
        raise KeyError(f"Unknown --task-spec {name!r}; have {sorted(TASK_SPECS)}.")
    return TASK_SPECS[name]()
