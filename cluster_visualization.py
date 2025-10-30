import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def visualize_clusters(G, embeddings, labels, zt=None, figsize=(12,8), title="Deep Zone Clustering"):
    plt.figure(figsize=figsize)
    if zt is not None and hasattr(zt, "get_zone_center"):
        pos = {n: zt.get_zone_center(n) for n in G.nodes()}
    else:
        pos = nx.spring_layout(G, seed=42)

    colors = plt.cm.tab10(labels / max(labels))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=800, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title(title)
    plt.axis("off")
    plt.show()