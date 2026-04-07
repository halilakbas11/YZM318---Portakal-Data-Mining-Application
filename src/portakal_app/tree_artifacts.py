from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from types import SimpleNamespace
from typing import Any

from portakal_app.data.models import DatasetHandle


DEFAULT_TREE_CLASS_COLORS = (
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
)


@dataclass
class TreeNodeArtifact:
    node_id: int
    sample_indices: list[int]
    child_ids: list[int] = field(default_factory=list)
    split_attr: str | None = None
    rules: tuple[str, ...] = ()
    branch_label: str = ""
    class_distribution: tuple[float, ...] = ()
    prediction: Any = None


@dataclass
class DecisionTreeArtifact:
    tree_id: str
    display_name: str
    instances: DatasetHandle
    root_id: int
    nodes: dict[int, TreeNodeArtifact]
    target_name: str
    kind: str
    class_values: tuple[str, ...] = ()
    class_colors: tuple[str, ...] = DEFAULT_TREE_CLASS_COLORS
    meta_target_class_index: int | None = None
    meta_size_calc_idx: int | None = None
    meta_depth_limit: int | None = None

    @property
    def root(self) -> int:
        return self.root_id

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def max_depth(self) -> int:
        def _depth(node_id: int) -> int:
            children = self.children(node_id)
            if not children:
                return 0
            return 1 + max(_depth(child_id) for child_id in children)

        return _depth(self.root_id) if self.nodes else 0

    @property
    def has_discrete_class(self) -> bool:
        return self.kind == "classification"

    def node(self, node_id: int) -> TreeNodeArtifact:
        return self.nodes[node_id]

    def children(self, node_id: int) -> list[int]:
        return list(self.nodes[node_id].child_ids)

    def is_leaf(self, node_id: int) -> bool:
        return not self.nodes[node_id].child_ids

    def weight(self, node_id: int) -> int:
        return max(1, len(self.nodes[node_id].sample_indices))

    def num_samples(self, node_id: int) -> int:
        return len(self.nodes[node_id].sample_indices)

    def get_distribution(self, node_id: int) -> list[list[float]]:
        return [list(self.nodes[node_id].class_distribution)]

    def attribute(self, node_id: int):
        name = self.nodes[node_id].split_attr or ""
        return SimpleNamespace(name=name)

    def rules(self, node_id: int) -> tuple[str, ...]:
        return self.nodes[node_id].rules

    def get_indices(self, node_ids: list[int] | tuple[int, ...] | set[int] | int) -> list[int]:
        if isinstance(node_ids, int):
            labels = [node_ids]
        else:
            labels = list(node_ids)
        indices: set[int] = set()
        for node_id in labels:
            node = self.nodes.get(int(node_id))
            if node is None:
                continue
            indices.update(int(index) for index in node.sample_indices)
        return sorted(indices)

    def reverse_children(self, node_id: int) -> None:
        self.nodes[node_id].child_ids = list(reversed(self.nodes[node_id].child_ids))

    def shuffle_children(self, random_state: Random | None = None) -> None:
        rng = random_state or Random()
        for node in self.nodes.values():
            if len(node.child_ids) > 1:
                rng.shuffle(node.child_ids)


@dataclass
class RandomForestArtifact:
    forest_id: str
    display_name: str
    instances: DatasetHandle
    trees: list[DecisionTreeArtifact]
    kind: str
    class_values: tuple[str, ...] = ()
    class_colors: tuple[str, ...] = DEFAULT_TREE_CLASS_COLORS

    @property
    def has_discrete_class(self) -> bool:
        return self.kind == "classification"

    @property
    def max_depth(self) -> int:
        return max((tree.max_depth for tree in self.trees), default=0)
