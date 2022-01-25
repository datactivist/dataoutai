import json
from pickle import load
from random import choice
from typing import Dict

import networkx as nx
import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (
    Scatter,
    MultiLine,
    NodesAndLinkedEdges,
    HoverTool,
    Select,
    Slider,
    TextInput,
    Label,
)
from bokeh.plotting import from_networkx, figure, curdoc
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics.cluster import (
    homogeneity_completeness_v_measure,
    contingency_matrix,
)
from sklearn.preprocessing import LabelEncoder

file_path = "../data/datasud.json"
TRAINING_METHOD_DEFAULT = "tfidf"
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
        self.producer = None


current_state = State()


def build_gold_groups_producers():
    with open(file_path, encoding="utf8") as f:
        datasets = json.load(f)
    datasets = datasets["datasets"]
    groups = set()
    ds_groups = dict()
    producer_nb_datasets = dict()
    for d in datasets:
        [groups.add(group) for group in d["metadata"]["groups"]]
        ds_groups[d["dataset_name"]] = d["metadata"]["groups"]
        producer_nb_datasets[d["author"]] = producer_nb_datasets.get("author", 0) + 1
    producer_list = list(
        dict(
            sorted(producer_nb_datasets.items(), key=lambda item: item[1], reverse=True)
        ).keys()
    )
    return groups, producer_list, ds_groups


gold_groups, producers, dataset_groups = build_gold_groups_producers()


def evaluate_clusters():
    gold = []
    pred = []
    le = LabelEncoder()
    all_group = list({x for group_list in dataset_groups.values() for x in group_list})
    le.fit(all_group)
    for dataset, groups in dataset_groups.items():
        if len(groups) == 1:
            gold.append(le.transform(groups)[0])
            pred.append(current_state.pred_groups[dataset])

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        gold, pred
    )
    fowlkes_mallows = fowlkes_mallows_score(gold, pred)
    contingency = contingency_matrix(gold, pred)
    purity = np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

    return {
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
        "fowlkes_mallows": fowlkes_mallows,
        "purity": purity,
    }


def build_file_name():
    return (
        f"../models/evaluation/clusters/"
        f"{current_state.method}_clusters_datasud_{current_state.clustering_method}_{current_state.nb_clusters}.pkl"
    )


def load_pickle_dump():
    with open(build_file_name(), "rb") as pickle_dump:
        current_state.pred_groups = load(pickle_dump)


plot = figure(
    title="Graph visualization",
    sizing_mode="stretch_both",
    # y_range=(-1.1, 1.1),
    # x_range=(-1.1, 1.1),
)


score_label = Label(
    x=0,
    y=0,
    x_units="screen",
    y_units="screen",
    text="",
)


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
    network_graph.node_renderer.glyph = Scatter(
        size="size", fill_color="color", marker="marker"
    )
    network_graph.edge_renderer.glyph = MultiLine(
        line_color="color", line_alpha=0.8, line_width=1
    )

    # Highlighting
    network_graph.node_renderer.hover_glyph = Scatter(
        fill_color="color", size="size", line_width=2, marker="marker"
    )
    network_graph.node_renderer.selection_glyph = Scatter(
        fill_color="color", size="size", line_width=2, marker="marker"
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

    """
    {
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
        "fowlkes_mallows": fowlkes_mallows,
        "purity": purity,
    }
    """
    results = evaluate_clusters()
    score_label.text = (
        f"Homogeneity: {results['homogeneity']:0.4f}\n"
        f"Completeness: {results['completeness']:0.4f}\n"
        f"V Measure: {results['v_measure']:0.4f}\n"
        f"Fowlkes Mallows: {results['fowlkes_mallows']:0.4f}\n"
        f"Purity: {results['purity']:0.4f}"
    )


nx_graph = nx.Graph()

plot.title.text = "Interactive Dataset Clustering Representation"

node_hover_tool = HoverTool(tooltips=[("Dataset name", "@name")])
plot.toolbar.active_scroll = "auto"
plot.add_tools(node_hover_tool)
plot.axis.visible = False
plot.add_layout(score_label)

training_method_select = Select(
    title="Select the method:",
    value=TRAINING_METHOD_DEFAULT,
    options=[
        ("fastTextfr", "Fastext"),
        ("fastText", "Fastext Facebook"),
        ("glove", "GloVe"),
        ("lsa", "LSA"),
        ("sbert", "sBert"),
        ("tfidf", "TFIDF"),
        ("word2vec", "Word2Vec"),
    ],
)


def training_method_callback(_, __, new):
    current_state.method = new
    draw()


training_method_select.on_change("value", training_method_callback)

nb_cluster_slider = Slider(start=5, end=30, step=1, value=NB_CLUSTER_DEFAULT)


def nb_cluster_callback(_, __, new):
    current_state.nb_clusters = new
    draw()


nb_cluster_slider.on_change("value", nb_cluster_callback)

highlight_select = Select(
    title="Select the group to highlight:",
    value=None,
    options=[None] + list(sorted(gold_groups)),
)


def highlight_callback(_, __, new):
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


def clustering_method_callback(_, __, new):
    current_state.clustering_method = new
    draw()


clustering_method_select.on_change("value", clustering_method_callback)

node_limit_field = TextInput(
    title="Select the maximum number of nodes (-1 to show all)",
    value="50",
)


def node_limit_callback(_, __, new):
    try:
        new = int(new)
        current_state.max_nodes = new
        draw()
    except ValueError:
        pass


node_limit_field.on_change("value", node_limit_callback)

producer_select = Select(
    title="Select the producer:",
    value=None,
    options=[None] + producers,
)


def producer_callback(_, __, new):
    current_state.producer = new
    draw()


producer_select.on_change("value", producer_callback)

curdoc().add_root(
    column(
        row(
            training_method_select,
            nb_cluster_slider,
            highlight_select,
            clustering_method_select,
            node_limit_field,
            producer_select,
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

        if pred_groups:
            actual_groups = [str(pred_groups[d["dataset_name"]])]
            dataset_gold_groups = d["metadata"]["groups"]

        else:
            dataset_gold_groups = []
            actual_groups = d["metadata"]["groups"]

        for actual_group in actual_groups:
            if actual_group not in groups:
                groups[actual_group] = [d["dataset_name"]]
            else:
                groups[actual_group].append(d["dataset_name"])

            size = 11
            node_color = None
            if highlight_gold in dataset_gold_groups:
                node_color = "#FFC0CB"
                size = 15

            shape = "square" if d["author"] == current_state.producer else "circle"

            nx_graph.add_node(
                d["dataset_name"],
                name=d["dataset_name"],
                size=size,
                color=node_color,
                gold=dataset_gold_groups,
                marker=shape,
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
            marker="circle",
        )
        for ind in indexes:
            nx_graph.add_edge(group, ind, color=group_color[group], weight=0.1)


draw()
