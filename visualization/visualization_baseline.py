import json
from random import choice
import networkx as nx

from bokeh.io import show, output_file
from bokeh.models import (
    Circle,
    MultiLine,
    NodesAndLinkedEdges,
    HoverTool,
)
from bokeh.plotting import from_networkx, figure

from pyvis.network import Network
from typing import Dict


class Visualizer:
    def __init__(self, file_path: str):
        self.nx_graph = nx.Graph()
        self.file_path = file_path

    def compute_graph(self, max_nodes: int = 50, pred_groups: Dict = None):
        """
        Compute the Networkx graph of the given file
        :param max_nodes: The max number of node displayed
        :param pred_groups: Use predicted group instead of "gold" group of the dataset
        """
        with open(self.file_path, encoding="utf8") as f:
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
            else:
                actual_groups = d["metadata"]["groups"]

            for actual_group in actual_groups:
                if actual_group not in groups:
                    groups[actual_group] = [d["dataset_name"]]
                else:
                    groups[actual_group].append(d["dataset_name"])
                self.nx_graph.add_node(d["dataset_name"], title=d["author"], size=7)

        group_color = {
            group: "#" + "".join([choice("0123456789ABCDEF") for _ in range(6)])
            for group in groups
        }
        for group, indexes in groups.items():
            c += 1
            self.nx_graph.add_node(
                group, title=group, color=group_color[group], size=15
            )
            for ind in indexes:
                self.nx_graph.add_edge(group, ind, color=group_color[group])

    def show_bokeh(self, name: str):
        """
        Display the graph using the bokeh library
        """
        plot = figure(
            title="Graph visualization",
            sizing_mode="scale_height",
            y_range=(-1.1, 1.1),
            x_range=(-1.1, 1.1),
        )
        plot.title.text = "Graph Interaction Demonstration"

        node_hover_tool = HoverTool(tooltips=[("Dataset name", "@dataset_name")])
        plot.toolbar.active_scroll = "auto"
        plot.add_tools(node_hover_tool)
        plot.axis.visible = False

        network_graph = from_networkx(self.nx_graph, nx.spring_layout, center=(0, 0))
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

        output_file(f"{name}.html")
        show(plot)

    def show_pyvis(self, name: str):
        """
        Display the graph using the pyvis library (way slower but have some nice physics and adaptative display of title according to the zoom)
        """
        network = Network(
            height="750px", width="100%", bgcolor="#222222", font_color="white"
        )
        network.from_nx(self.nx_graph)
        network.set_options(
            """
                var options = {
          "physics": {
            "forceAtlas2Based": {
              "springLength": 130
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
          }
        }
        """
        )
        # To be able to configure with a GUI a configuration like the one above
        # self.network.show_buttons(filter_=["physics"])
        network.show(f"{name}.html")
