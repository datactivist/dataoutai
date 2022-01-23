import json
from pickle import load
from random import choice
from typing import Dict

import networkx as nx
from bokeh.layouts import column, row
from bokeh.models import (
    Circle,
    MultiLine,
    NodesAndLinkedEdges,
    HoverTool,
    Select,
    Slider,
    TextInput,
)
from bokeh.plotting import from_networkx, figure, curdoc

file_path = "../data/datasud.json"
TRAINING_METHOD_DEFAULT = "TFIDF"
NB_CLUSTER_DEFAULT = 15
DEFAULT_CLUSTERING_METHOD = "km"


class State:
    def __init__(self):
        self.max_nodes = 50
        self.pred_groups = None
        self.highlight_gold = None
        self.method = TRAINING_METHOD_DEFAULT
        self.nb_clusters = NB_CLUSTER_DEFAULT
        self.clustering_method = DEFAULT_CLUSTERING_METHOD


current_state = State()


def build_gold_group_list():
    with open(file_path, encoding="utf8") as f:
        datasets = json.load(f)
    datasets = datasets["datasets"]
    groups = set()
    for d in datasets:
        [groups.add(group) for group in d["metadata"]["groups"]]
    return groups


def load_pickle_dump():
    with open(
        f"../models/evaluation/clusters/{current_state.method.lower()}_clusters_datasud_{current_state.clustering_method}_{current_state.nb_clusters}.pkl",
        "rb",
    ) as pickle_dump:
        current_state.pred_groups = load(pickle_dump)


def draw():
    plot.renderers = []
    nx_graph.clear()
    load_pickle_dump()
    compute_graph(
        current_state.max_nodes, current_state.pred_groups, current_state.highlight_gold
    )

    network_graph = from_networkx(
        nx_graph,
        nx.spring_layout(nx_graph, scale=2),
        center=(0, 0),
    )
    network_graph.node_renderer.glyph = Circle(size="size", fill_color="color")
    network_graph.edge_renderer.glyph = MultiLine(
        line_color="color", line_alpha=0.8, line_width=1
    )

    # Highlighting
    network_graph.node_renderer.hover_glyph = Circle(
        fill_color="color", size="size", line_width=2
    )
    network_graph.node_renderer.selection_glyph = Circle(
        fill_color="color", size="size", line_width=2
    )

    network_graph.edge_renderer.selection_glyph = MultiLine(
        line_color="color", line_width=2
    )
    network_graph.edge_renderer.hover_glyph = MultiLine(
        line_color="color", line_width=2
    )

    network_graph.selection_policy = NodesAndLinkedEdges()
    network_graph.inspection_policy = NodesAndLinkedEdges()
    plot.renderers.append(network_graph)


nx_graph = nx.Graph()
gold_groups = build_gold_group_list()

plot = figure(
    title="Graph visualization",
    sizing_mode="scale_both",
    y_range=(-1.1, 1.1),
    x_range=(-1.1, 1.1),
)
plot.title.text = "Graph Interaction Demonstration"

node_hover_tool = HoverTool(tooltips=[("Dataset name", "@name")])
plot.toolbar.active_scroll = "auto"
plot.add_tools(node_hover_tool)
plot.axis.visible = False

training_method_select = Select(
    title="Select the method:",
    value=TRAINING_METHOD_DEFAULT,
    options=["LSA", "sBert", "TFIDF", "Word2Vec"],
)


def training_method_callback(_, old, new):
    current_state.method = new
    draw()


training_method_select.on_change("value", training_method_callback)

nb_cluster_slider = Slider(start=5, end=30, step=1, value=NB_CLUSTER_DEFAULT)


def nb_cluster_callback(_, old, new):
    current_state.nb_clusters = new
    draw()


nb_cluster_slider.on_change("value", nb_cluster_callback)

highlight_select = Select(
    title="Select the group to highlight:",
    value=None,
    options=[None] + list(sorted(gold_groups)),
)


def highlight_callback(_, old, new):
    current_state.highlight_gold = new
    draw()


highlight_select.on_change("value", highlight_callback)

clustering_method_select = Select(
    title="Select the clustering method:",
    value=DEFAULT_CLUSTERING_METHOD,
    options=[
        ("ac", "Agglomerative Clustering"),
        ("km", "KMeans"),
        ("sc", "Spectral Clustering"),
    ],
)


def clustering_method_callback(_, old, new):
    current_state.clustering_method = new
    draw()


clustering_method_select.on_change("value", clustering_method_callback)

node_limit_field = TextInput(
    title="Select the maximum number of nodes (-1 to show all)",
    value="50",
)


def node_limit_callback(_, old, new):
    try:
        new = int(new)
        current_state.max_nodes = new
        draw()
    except ValueError:
        pass


node_limit_field.on_change("value", node_limit_callback)


curdoc().add_root(
    column(
        row(
            training_method_select,
            nb_cluster_slider,
            highlight_select,
            clustering_method_select,
            node_limit_field,
        ),
        plot,
    )
)


def compute_graph(
    max_nodes: int = 50, pred_groups: Dict = None, highlight_gold: str = None
):
    """
    Compute the Networkx graph of the given file
    :param max_nodes: The max number of node displayed (-1 for all)
    :param pred_groups: Use predicted group instead of "gold" group of the dataset
    :param highlight_gold: group to highlight
    """
    with open(file_path, encoding="utf8") as f:
        datas = json.load(f)

    datasets = datas["datasets"]

    if max_nodes == -1 or max_nodes > len(datasets):
        max_nodes = len(datasets)
    c = 0
    groups = {}
    for d in datasets:
        if c > max_nodes:
            break

        c += 1
        # [gold_groups.add(group) for group in d["metadata"]["groups"]]

        if pred_groups:
            actual_groups = [str(pred_groups[d["dataset_name"]])]
            gold_groups = d["metadata"]["groups"]

        else:
            gold_groups = []
            actual_groups = d["metadata"]["groups"]

        for actual_group in actual_groups:
            if actual_group not in groups:
                groups[actual_group] = [d["dataset_name"]]
            else:
                groups[actual_group].append(d["dataset_name"])

            size = 11
            node_color = None
            if highlight_gold in gold_groups:
                node_color = "#FFC0CB"
                size = 15

            nx_graph.add_node(
                d["dataset_name"],
                name=d["dataset_name"],
                size=size,
                color=node_color,
                gold=gold_groups,
            )

    group_color = {
        group: "#" + "".join([choice("0123456789ABCDEF") for _ in range(6)])
        for group in groups
    }
    for group, indexes in groups.items():
        c += 1
        nx_graph.add_node(
            group,
            name=group,
            color=group_color[group],
            size=15,
            gold=[],
        )
        for ind in indexes:
            nx_graph.add_edge(group, ind, color=group_color[group], weight=0.1)


draw()
