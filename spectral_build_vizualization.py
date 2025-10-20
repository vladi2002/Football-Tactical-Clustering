import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from collections import defaultdict

from graph import ZoneTransformer

zt = ZoneTransformer()

def build_enriched_similarity_matrix(G, alpha=0.4, beta=0.3, gamma=0.3):
    """
    Build a similarity matrix that combines:
    - Structural connectivity (edges)
    - Edge features (transition types, frequencies)
    - Node features (spatial, event-based)
    
    This captures which zones "belong together" tactically.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    
    # === 1. STRUCTURAL SIMILARITY (Adjacency) ===
    G_undirected = G.to_undirected()
    A = nx.to_numpy_array(G_undirected, nodelist=nodes, weight="weight")
    A_norm = A / (A.max() if A.max() > 0 else 1)
    
    # === 2. EDGE FEATURE SIMILARITY ===
    # Zones that have similar transition patterns are tactically similar
    
    # 2a. Transition frequency similarity
    A_freq = np.zeros((n, n))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            freq = 0
            if G.has_edge(u, v):
                freq += G[u][v].get("transition_frequency", 0)
            if G.has_edge(v, u):
                freq += G[v][u].get("transition_frequency", 0)
            A_freq[i, j] = freq / 2
    
    # 2b. Edge event type similarity
    # Build feature vectors for each node based on its edge event types
    from collections import Counter
    node_edge_features = []
    
    for node in nodes:
        # Get all edges connected to this node
        in_edges = list(G.in_edges(node, data=True))
        out_edges = list(G.out_edges(node, data=True))
        all_edges = in_edges + out_edges
        
        if all_edges:
            # Count event types
            event_types = [e[2].get("most_common_event", "UNKNOWN") for e in all_edges]
            event_counter = Counter(event_types)
            total = len(all_edges)
            
            # Transition characteristics
            avg_weight = np.mean([e[2].get("weight", 0) for e in all_edges])
            avg_freq = np.mean([e[2].get("transition_frequency", 0) for e in all_edges])
            
            feature_vec = [
                event_counter.get("PASS", 0) / total,
                event_counter.get("SHOT", 0) / total,
                event_counter.get("DUEL", 0) / total,
                event_counter.get("GOALKEEPER", 0) / total,
                avg_weight / 100,  # normalize
                avg_freq,
            ]
        else:
            feature_vec = [0, 0, 0, 0, 0, 0]
        
        node_edge_features.append(feature_vec)
    
    X_edge = np.array(node_edge_features)
    S_edge = rbf_kernel(X_edge, gamma=0.5)
    
    # === 3. NODE FEATURE SIMILARITY ===
    # Zones with similar spatial/tactical properties are similar
    node_features = []
    
    for node in nodes:
        x, y = zt.get_zone_center(node)
        
        features = [
            x / 100,  # normalized x position
            y / 100,  # normalized y position
            G.nodes[node].get("event_count", 0) / 100,  # normalize
            G.nodes[node].get("unique_players", 0) / 11,  # normalize by team size
            *G.nodes[node].get("role_distribution", [0, 0, 0, 0]),  # [GK, DF, MD, FW]
        ]
        node_features.append(features)
    
    X_node = np.array(node_features)
    X_node = StandardScaler().fit_transform(X_node)
    S_node = rbf_kernel(X_node, gamma=0.5)
    
    # === 4. COMBINE ALL SIMILARITIES ===
    S_combined = alpha * A_norm + beta * S_edge + gamma * S_node
    
    # Ensure symmetry
    S_combined = (S_combined + S_combined.T) / 2
    
    return S_combined, nodes


def discover_tactical_patterns(G, k=5, alpha=0.4, beta=0.3, gamma=0.3):
    """
    Discover tactical patterns (zone clusters) in a match.
    
    Each pattern represents a tactical "module" - a group of zones that are
    frequently connected and represent a coherent tactical structure.
    
    Args:
        G: Zone transition graph for a match/team
        k: Number of tactical patterns to discover
        alpha, beta, gamma: Weights for structural, edge, node similarities
    
    Returns:
        labels: Cluster assignment for each zone
        nodes: List of zone IDs
    """
    
    # Build similarity matrix
    S_combined, nodes = build_enriched_similarity_matrix(G, alpha, beta, gamma)
    
    # Spectral clustering
    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42
    )
    labels = sc.fit_predict(S_combined)
    
    # Assign cluster labels back to graph
    for node, label in zip(nodes, labels):
        G.nodes[node]["tactical_pattern"] = int(label)
    
    return labels, nodes


def interpret_tactical_patterns(G, labels, nodes, k=5):
    """
    Interpret what each tactical pattern represents.
    """
    print("\n" + "="*80)
    print("TACTICAL PATTERN DISCOVERY")
    print("="*80)
    
    for pattern_id in range(k):
        pattern_zones = [nodes[i] for i in range(len(nodes)) if labels[i] == pattern_id]
        
        if not pattern_zones:
            continue
        
        print(f"\n{'='*80}")
        print(f"PATTERN {pattern_id}: {len(pattern_zones)} zones")
        print(f"{'='*80}")
        
        # Zone names
        zone_names = [G.nodes[z]["zone_name"] for z in pattern_zones]
        print(f"Zones: {pattern_zones}")
        print(f"Names: {zone_names[:5]}{'...' if len(zone_names) > 5 else ''}")
        
        # Spatial characteristics
        zone_centers = [zt.get_zone_center(z) for z in pattern_zones]
        avg_x = np.mean([c[0] for c in zone_centers])
        avg_y = np.mean([c[1] for c in zone_centers])
        spread_x = np.std([c[0] for c in zone_centers])
        spread_y = np.std([c[1] for c in zone_centers])
        
        print(f"\nSpatial profile:")
        print(f"  • Center: ({avg_x:.1f}, {avg_y:.1f})")
        print(f"  • Spread: X={spread_x:.1f}, Y={spread_y:.1f}")
        
        # Determine spatial region
        if avg_x < 33:
            x_region = "DEFENSIVE"
        elif avg_x < 67:
            x_region = "MIDFIELD"
        else:
            x_region = "ATTACKING"
        
        if spread_y > 25:
            y_region = "WIDE"
        elif avg_y < 35 or avg_y > 65:
            y_region = "FLANK"
        else:
            y_region = "CENTRAL"
        
        print(f"  • Region: {y_region} {x_region}")
        
        # Internal connectivity
        internal_edges = [(u, v) for u in pattern_zones for v in pattern_zones 
                         if u != v and G.has_edge(u, v)]
        external_edges = [(u, v) for u in pattern_zones 
                         for v in nodes if v not in pattern_zones and G.has_edge(u, v)]
        
        print(f"\nConnectivity:")
        print(f"  • Internal edges: {len(internal_edges)}")
        print(f"  • External edges: {len(external_edges)}")
        
        if internal_edges:
            internal_weight = sum(G[u][v]["weight"] for u, v in internal_edges)
            print(f"  • Internal flow strength: {internal_weight}")
        
        # Event composition
        total_events = sum(G.nodes[z]["event_count"] for z in pattern_zones)
        print(f"  • Total events: {total_events}")
        
        # Role distribution
        avg_roles = np.mean([G.nodes[z]["role_distribution"] for z in pattern_zones], axis=0)
        print(f"\nPlayer roles:")
        print(f"  • GK: {avg_roles[0]:.2%}, DF: {avg_roles[1]:.2%}, MD: {avg_roles[2]:.2%}, FW: {avg_roles[3]:.2%}")
        
        # Suggest tactical interpretation
        print(f"\n📋 Interpretation:", end=" ")
        if x_region == "ATTACKING" and avg_roles[3] > 0.4:
            print("FINAL THIRD ATTACKING PATTERN")
        elif x_region == "DEFENSIVE" and avg_roles[1] > 0.4:
            print("DEFENSIVE BUILD-UP PATTERN")
        elif y_region == "WIDE" or y_region == "FLANK":
            print(f"{y_region} PLAY PATTERN (wing-based)")
        elif y_region == "CENTRAL" and x_region == "MIDFIELD":
            print("CENTRAL MIDFIELD PATTERN (possession/progression)")
        elif len(internal_edges) > 0 and internal_weight / len(internal_edges) > 10:
            print("HIGH-INTENSITY TRANSITION PATTERN")
        else:
            print("GENERAL PLAY PATTERN")


def compare_tactical_consistency(graphs, team_ids, k=5):
    """
    Compare tactical patterns across multiple matches for the same team.
    This tests whether patterns are consistent (as they should be for tactical approaches).
    
    Args:
        graphs: List of (graph, team_id) tuples
        team_ids: Unique team IDs to analyze
        k: Number of patterns
    
    Returns:
        consistency_scores: Dict mapping team_id to consistency score
    """
    print("\n" + "="*80)
    print("TACTICAL CONSISTENCY ANALYSIS")
    print("="*80)
    
    team_patterns = defaultdict(list)
    
    # Discover patterns for each match
    for G, team_id in graphs:
        labels, nodes = discover_tactical_patterns(G, k=k)
        team_patterns[team_id].append((labels, nodes))
    
    # Measure consistency
    consistency_scores = {}
    
    for team_id in team_ids:
        if team_id not in team_patterns or len(team_patterns[team_id]) < 2:
            continue
        
        patterns = team_patterns[team_id]
        
        # Compare pattern assignments across matches using Adjusted Rand Index
        rand_scores = []
        for i in range(len(patterns)):
            for j in range(i+1, len(patterns)):
                labels_i, nodes_i = patterns[i]
                labels_j, nodes_j = patterns[j]
                
                # Only compare zones that appear in both matches
                common_nodes = set(nodes_i) & set(nodes_j)
                if len(common_nodes) < 5:  # Need enough overlap
                    continue
                
                # Extract labels for common nodes
                idx_i = [nodes_i.index(n) for n in common_nodes]
                idx_j = [nodes_j.index(n) for n in common_nodes]
                common_labels_i = [labels_i[i] for i in idx_i]
                common_labels_j = [labels_j[j] for j in idx_j]
                
                # Compute similarity
                ari = adjusted_rand_score(common_labels_i, common_labels_j)
                rand_scores.append(ari)
        
        if rand_scores:
            consistency_scores[team_id] = np.mean(rand_scores)
            print(f"\nTeam {team_id}:")
            print(f"  • Matches analyzed: {len(patterns)}")
            print(f"  • Average consistency (ARI): {consistency_scores[team_id]:.3f}")
            print(f"  • Interpretation: ", end="")
            if consistency_scores[team_id] > 0.5:
                print("HIGHLY CONSISTENT tactical approach")
            elif consistency_scores[team_id] > 0.3:
                print("MODERATELY CONSISTENT tactical approach")
            else:
                print("VARIABLE tactical approach (adapts to opponents)")
    
    return consistency_scores


def visualize_patterns_on_pitch(G, labels, nodes):
    """
    Visualize discovered tactical patterns on the pitch.
    """
    import matplotlib.patches as patches
    
    k = len(set(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, k))
    
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='#1a1a1a')
    ax.set_facecolor('#2d5016')
    
    # Draw zones colored by pattern
    for i, (node, label) in enumerate(zip(nodes, labels)):
        x_min, x_max, y_min, y_max = zt.get_zone_bounds(node)
        
        zone_rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor='white',
            facecolor=colors[label],
            alpha=0.6,
            zorder=1
        )
        ax.add_patch(zone_rect)
        
        # Zone ID
        x_center, y_center = zt.get_zone_center(node)
        ax.text(x_center, y_center, f"{node}\nP{label}",
               ha='center', va='center', fontsize=9,
               fontweight='bold', color='black',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Draw edges (colored by pattern)
    for u, v, data in G.edges(data=True):
        u_label = G.nodes[u]["tactical_pattern"]
        v_label = G.nodes[v]["tactical_pattern"]
        
        u_pos = zt.get_zone_center(u)
        v_pos = zt.get_zone_center(v)
        
        # Internal pattern edges are thicker
        if u_label == v_label:
            linewidth = 2 + data["weight"] / 10
            alpha = 0.8
            color = colors[u_label]
        else:
            linewidth = 1
            alpha = 0.3
            color = 'gray'
        
        ax.arrow(u_pos[0], u_pos[1],
                v_pos[0] - u_pos[0], v_pos[1] - u_pos[1],
                head_width=2, head_length=2, fc=color, ec=color,
                alpha=alpha, linewidth=linewidth, zorder=2)
    
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title('Tactical Patterns (Zone Clusters)', 
             fontsize=16, color='white', fontweight='bold', pad=20)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], edgecolor='white', 
                            label=f'Pattern {i}') for i in range(k)]
    ax.legend(handles=legend_elements, loc='upper left', 
             framealpha=0.8, fontsize=10)
    
    plt.tight_layout()
    plt.show()

