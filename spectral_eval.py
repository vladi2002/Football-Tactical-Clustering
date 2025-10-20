import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from kloppy import wyscout
from kloppy.domain import Orientation
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score
from collections import defaultdict
import seaborn as sns
from spectral_build_vizualization import discover_tactical_patterns, interpret_tactical_patterns, visualize_patterns_on_pitch

# Import your existing functions
from graph import (
    ZoneTransformer, 
    build_transition_graph
)

from graph import ZoneTransformer

zt = ZoneTransformer()

# ========================================
# 1. DATA PREPARATION
# ========================================

def load_match_graphs(match_ids, team_filter='both', time_window=None):
    """
    Load multiple matches and build graphs.
    
    Args:
        match_ids: List of Wyscout match IDs
        team_filter: 'home', 'away', or 'both'
        time_window: None (full match) or int (minutes per window)
    
    Returns:
        List of (graph, metadata) tuples
    """
    zt = ZoneTransformer()
    graphs = []
    
    INCLUDE_EVENTS = ["PASS", "SHOT", "DUEL", "GOALKEEPER"]
    
    for match_id in match_ids:
        print(f"\nLoading match {match_id}...")
        
        try:
            # Load Wyscout data
            data = wyscout.load_open_data(match_id)
            home_team, away_team = data.metadata.teams
            home_team_id = int(home_team.team_id)
            away_team_id = int(away_team.team_id)
            
            # Transform to attacking orientation
            data = data.transform(Orientation.ACTION_EXECUTING_TEAM)
            df_raw = data.to_df(*zt.INCLUDE_COLS, engine="pandas")
            df_raw.set_index("event_id", inplace=True)
            df_raw = zt.transform(df_raw)
            
            # Filter events
            df = df_raw[df_raw["event_type"].isin(INCLUDE_EVENTS)]
            
            if time_window is None:
                # Full match graphs
                if team_filter in ['home', 'both']:
                    df_home = df[df["team_id"] == home_team_id]
                    if len(df_home) > 10:  # Minimum events threshold
                        G_home = build_transition_graph(df_home)
                        graphs.append((G_home, {
                            'match_id': match_id,
                            'team_id': home_team_id,
                            'team_name': home_team.name,
                            'is_home': True,
                            'time_window': 'full'
                        }))
                
                if team_filter in ['away', 'both']:
                    df_away = df[df["team_id"] == away_team_id]
                    if len(df_away) > 10:
                        G_away = build_transition_graph(df_away)
                        graphs.append((G_away, {
                            'match_id': match_id,
                            'team_id': away_team_id,
                            'team_name': away_team.name,
                            'is_home': False,
                            'time_window': 'full'
                        }))
            else:
                # Time-windowed graphs
                max_time = df['timestamp'].max()
                
                for t_start in range(0, int(max_time), time_window * 60):
                    t_end = t_start + time_window * 60
                    df_window = df[(df['timestamp'] >= t_start) & 
                                  (df['timestamp'] < t_end)]
                    
                    if team_filter in ['home', 'both']:
                        df_home_window = df_window[df_window["team_id"] == home_team_id]
                        if len(df_home_window) > 5:
                            G_home = build_transition_graph(df_home_window)
                            graphs.append((G_home, {
                                'match_id': match_id,
                                'team_id': home_team_id,
                                'team_name': home_team.name,
                                'is_home': True,
                                'time_window': f"{t_start//60}-{t_end//60}min"
                            }))
                    
                    if team_filter in ['away', 'both']:
                        df_away_window = df_window[df_window["team_id"] == away_team_id]
                        if len(df_away_window) > 5:
                            G_away = build_transition_graph(df_away_window)
                            graphs.append((G_away, {
                                'match_id': match_id,
                                'team_id': away_team_id,
                                'team_name': away_team.name,
                                'is_home': False,
                                'time_window': f"{t_start//60}-{t_end//60}min"
                            }))
        
        except Exception as e:
            print(f"  Error loading match {match_id}: {e}")
            continue
    
    print(f"\nLoaded {len(graphs)} graphs total")
    return graphs


# ========================================
# 2. EVALUATION METRICS
# ========================================

def evaluate_clustering_quality(G, labels, nodes):
    """
    Compute clustering quality metrics (from proposal: Silhouette, Calinski-Harabasz).
    """
    from sklearn.preprocessing import StandardScaler
    
    # Build feature matrix for evaluation
    node_features = []
    for node in nodes:
        x, y = zt.get_zone_center(node)
        features = [
            x / 100,
            y / 100,
            G.nodes[node].get("event_count", 0),
            G.nodes[node].get("unique_players", 0),
            *G.nodes[node].get("role_distribution", [0, 0, 0, 0]),
        ]
        node_features.append(features)
    
    X = np.array(node_features)
    X = StandardScaler().fit_transform(X)
    
    # Compute metrics
    if len(set(labels)) > 1 and len(labels) > len(set(labels)):
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
    else:
        silhouette = -1
        calinski = -1
    
    return {
        'silhouette': silhouette,
        'calinski_harabasz': calinski
    }


def evaluate_team_consistency(graphs_by_team, k=5):
    """
    Measure consistency of patterns across matches for each team (using ARI).
    """
    consistency_results = {}
    
    for team_id, team_graphs in graphs_by_team.items():
        if len(team_graphs) < 2:
            continue
        
        team_name = team_graphs[0][1]['team_name']
        
        # Discover patterns for each match
        patterns = []
        for G, metadata in team_graphs:
            labels, nodes = discover_tactical_patterns(G, k=k)
            patterns.append((labels, nodes, metadata))
        
        # Compute pairwise ARI
        ari_scores = []
        for i in range(len(patterns)):
            for j in range(i+1, len(patterns)):
                labels_i, nodes_i, _ = patterns[i]
                labels_j, nodes_j, _ = patterns[j]
                
                # Find common nodes
                common_nodes = list(set(nodes_i) & set(nodes_j))
                if len(common_nodes) < 5:
                    continue
                
                # Extract labels for common nodes
                idx_i = [nodes_i.index(n) for n in common_nodes]
                idx_j = [nodes_j.index(n) for n in common_nodes]
                common_labels_i = [labels_i[idx] for idx in idx_i]
                common_labels_j = [labels_j[idx] for idx in idx_j]
                
                ari = adjusted_rand_score(common_labels_i, common_labels_j)
                ari_scores.append(ari)
        
        if ari_scores:
            consistency_results[team_id] = {
                'team_name': team_name,
                'num_matches': len(team_graphs),
                'mean_ari': np.mean(ari_scores),
                'std_ari': np.std(ari_scores),
                'min_ari': np.min(ari_scores),
                'max_ari': np.max(ari_scores)
            }
    
    return consistency_results


# ========================================
# 3. EXPERIMENTS (from proposal Table 1)
# ========================================

def experiment_1_parameter_sensitivity(graphs, k_range=[3, 4, 5, 6, 7]):
    """
    Experiment 1: Test different numbers of clusters (k).
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: PARAMETER SENSITIVITY (k)")
    print("="*80)
    
    results = []
    
    for k in k_range:
        print(f"\nTesting k={k}...")
        
        silhouettes = []
        calinski_scores = []
        
        for G, metadata in graphs[:10]:  # Sample 10 graphs for speed
            labels, nodes = discover_tactical_patterns(G, k=k)
            metrics = evaluate_clustering_quality(G, labels, nodes)
            
            if metrics['silhouette'] > -1:
                silhouettes.append(metrics['silhouette'])
                calinski_scores.append(metrics['calinski_harabasz'])
        
        if silhouettes:
            results.append({
                'k': k,
                'mean_silhouette': np.mean(silhouettes),
                'std_silhouette': np.std(silhouettes),
                'mean_calinski': np.mean(calinski_scores),
                'std_calinski': np.std(calinski_scores)
            })
            
            print(f"  Silhouette: {np.mean(silhouettes):.3f} ± {np.std(silhouettes):.3f}")
            print(f"  Calinski-Harabasz: {np.mean(calinski_scores):.1f} ± {np.std(calinski_scores):.1f}")
    
    # Plot results
    df_results = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].errorbar(df_results['k'], df_results['mean_silhouette'], 
                     yerr=df_results['std_silhouette'], marker='o', capsize=5)
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title('Cluster Quality vs k')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].errorbar(df_results['k'], df_results['mean_calinski'], 
                     yerr=df_results['std_calinski'], marker='o', capsize=5)
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Calinski-Harabasz Index')
    axes[1].set_title('Separation Quality vs k')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment1_k_sensitivity.png', dpi=150)
    plt.show()
    
    return df_results


def experiment_2_similarity_weights(graphs, k=5):
    """
    Experiment 2: Test different combinations of similarity weights (α, β, γ).
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: SIMILARITY WEIGHT COMBINATIONS")
    print("="*80)
    
    # Test different weight combinations
    weight_configs = [
        {'name': 'Structure-only', 'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0},
        {'name': 'Structure+Edge', 'alpha': 0.6, 'beta': 0.4, 'gamma': 0.0},
        {'name': 'Structure+Node', 'alpha': 0.6, 'beta': 0.0, 'gamma': 0.4},
        {'name': 'Balanced', 'alpha': 0.4, 'beta': 0.3, 'gamma': 0.3},
        {'name': 'Edge-focused', 'alpha': 0.2, 'beta': 0.6, 'gamma': 0.2},
    ]
    
    results = []
    
    for config in weight_configs:
        print(f"\nTesting: {config['name']}")
        print(f"  α={config['alpha']}, β={config['beta']}, γ={config['gamma']}")
        
        silhouettes = []
        
        for G, metadata in graphs[:10]:
            labels, nodes = discover_tactical_patterns(
                G, k=k, 
                alpha=config['alpha'], 
                beta=config['beta'], 
                gamma=config['gamma']
            )
            metrics = evaluate_clustering_quality(G, labels, nodes)
            
            if metrics['silhouette'] > -1:
                silhouettes.append(metrics['silhouette'])
        
        if silhouettes:
            results.append({
                'config': config['name'],
                'alpha': config['alpha'],
                'beta': config['beta'],
                'gamma': config['gamma'],
                'mean_silhouette': np.mean(silhouettes),
                'std_silhouette': np.std(silhouettes)
            })
            
            print(f"  Silhouette: {np.mean(silhouettes):.3f} ± {np.std(silhouettes):.3f}")
    
    # Visualize
    df_results = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    plt.barh(df_results['config'], df_results['mean_silhouette'], 
             xerr=df_results['std_silhouette'], capsize=5)
    plt.xlabel('Silhouette Score')
    plt.title('Impact of Similarity Weight Configuration')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('experiment2_weights.png', dpi=150)
    plt.show()
    
    return df_results


def experiment_3_team_consistency(graphs, k=5):
    """
    Experiment 3: Measure tactical consistency across matches for each team.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: TEAM TACTICAL CONSISTENCY")
    print("="*80)
    
    # Group by team
    graphs_by_team = defaultdict(list)
    for G, metadata in graphs:
        graphs_by_team[metadata['team_id']].append((G, metadata))
    
    # Evaluate consistency
    consistency_results = evaluate_team_consistency(graphs_by_team, k=k)
    
    # Print results
    df_consistency = pd.DataFrame(consistency_results.values())
    # Guard: handle case with no results (e.g., <2 matches per team)
    if df_consistency.empty or 'mean_ari' not in df_consistency.columns:
        print("\nNo team consistency results (need ≥2 matches per team). Skipping Experiment 3 charts.")
        return pd.DataFrame()
    df_consistency = df_consistency.sort_values('mean_ari', ascending=False)
    
    print("\n" + "-"*80)
    print(f"{'Team':<30} {'Matches':>8} {'Mean ARI':>10} {'Std ARI':>10}")
    print("-"*80)
    for _, row in df_consistency.iterrows():
        print(f"{row['team_name']:<30} {row['num_matches']:>8} "
              f"{row['mean_ari']:>10.3f} {row['std_ari']:>10.3f}")
    
    # Visualize
    plt.figure(figsize=(12, 6))
    plt.barh(df_consistency['team_name'], df_consistency['mean_ari'], 
             xerr=df_consistency['std_ari'], capsize=5)
    plt.xlabel('Adjusted Rand Index (ARI)')
    plt.title('Tactical Consistency Across Matches by Team')
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, 
                label='High Consistency Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('experiment3_consistency.png', dpi=150)
    plt.show()
    
    return df_consistency


def experiment_4_pattern_interpretation(graphs, k=5, num_examples=3):
    """
    Experiment 4: Visualize and interpret discovered patterns.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 4: PATTERN INTERPRETATION")
    print("="*80)
    
    # Select diverse examples
    examples = graphs[:num_examples]
    
    for i, (G, metadata) in enumerate(examples):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i+1}: {metadata['team_name']} (Match {metadata['match_id']})")
        print(f"{'='*80}")
        
        labels, nodes = discover_tactical_patterns(G, k=k)
        interpret_tactical_patterns(G, labels, nodes, k=k)
        visualize_patterns_on_pitch(G, labels, nodes)


# ========================================
# 4. MAIN TESTING PIPELINE
# ========================================

def run_full_test_suite(match_ids=None, k=5):
    """
    Run complete test suite as described in the project proposal.
    """
    print("="*80)
    print("TACTICAL PATTERN DISCOVERY - FULL TEST SUITE")
    print("="*80)
    
    # Default to some matches if none provided
    if match_ids is None:
        match_ids = [2058017, 2058019, 2058020]  # Example Wyscout match IDs
    
    # Load data
    print("\n[1/5] Loading match data...")
    graphs = load_match_graphs(match_ids, team_filter='both', time_window=None)
    
    if len(graphs) < 2:
        print("ERROR: Not enough graphs loaded. Check match IDs.")
        return
    
    # Run experiments
    print("\n[2/5] Running Experiment 1: Parameter Sensitivity...")
    exp1_results = experiment_1_parameter_sensitivity(graphs, k_range=[3, 4, 5, 6, 7])
    
    print("\n[3/5] Running Experiment 2: Similarity Weights...")
    exp2_results = experiment_2_similarity_weights(graphs, k=k)
    
    print("\n[4/5] Running Experiment 3: Team Consistency...")
    exp3_results = experiment_3_team_consistency(graphs, k=k)
    
    print("\n[5/5] Running Experiment 4: Pattern Interpretation...")
    experiment_4_pattern_interpretation(graphs, k=k, num_examples=3)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)
    print(f"Total graphs analyzed: {len(graphs)}")
    print(f"Results saved:")
    print("  - experiment1_k_sensitivity.png")
    print("  - experiment2_weights.png")
    print("  - experiment3_consistency.png")
    print("  - Pattern visualizations displayed")
    
    return {
        'exp1': exp1_results,
        'exp2': exp2_results,
        'exp3': exp3_results,
        'graphs': graphs
    }


# # ========================================
# # 5. QUICK START
# # ========================================

# if __name__ == "__main__":
#     print("""
#     ========================================================================
#     QUICK START GUIDE
#     ========================================================================
    
#     Option 1 - Run full test suite:
#         results = run_full_test_suite(match_ids=[2058017, 2058019, 2058020])
    
#     Option 2 - Test single match:
#         graphs = load_match_graphs([2058017], team_filter='home')
#         G, metadata = graphs[0]
#         labels, nodes = discover_tactical_patterns(G, k=5)
#         interpret_tactical_patterns(G, labels, nodes, k=5)
#         visualize_patterns_on_pitch(G, labels, nodes)
    
#     Option 3 - Compare teams:
#         graphs = load_match_graphs([2058017], team_filter='both')
#         G_home, _ = graphs[0]
#         G_away, _ = graphs[1]
        
#         labels_home, nodes_home = discover_tactical_patterns(G_home, k=5)
#         labels_away, nodes_away = discover_tactical_patterns(G_away, k=5)
        
#         interpret_tactical_patterns(G_home, labels_home, nodes_home, k=5)
#         interpret_tactical_patterns(G_away, labels_away, nodes_away, k=5)
    
#     ========================================================================
#     """)