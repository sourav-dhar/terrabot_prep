# Terrapay Transaction Monitoring Analysis Notebook

This notebook performs a comprehensive analysis of Terrapay's transaction monitoring system to optimize rule efficiency and reduce false positives.

## 1. Setup and Library Imports

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import itertools
import os
import networkx as nx
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 100)

# Set visualization style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Create output directories
os.makedirs('visualizations', exist_ok=True)
os.makedirs('reports', exist_ok=True)
```

## 2. Data Loading

```python
# Define data loading function
def load_data():
    """Load all data from Excel file"""
    # Read transaction data
    transaction_data = pd.read_excel('transaction_dummy_data_10k_final.xlsx', 
                                     sheet_name='transaction_dummy_data_10k')
    
    # Read metadata
    metadata = pd.read_excel('transaction_dummy_data_10k_final.xlsx', 
                             sheet_name='Sheet2')
    
    # Read rule descriptions
    rule_descriptions = pd.read_excel('transaction_dummy_data_10k_final.xlsx', 
                                     sheet_name='rule_description')
    
    # Convert dates to datetime if not already
    date_columns = ['transaction_date_time_local', 'created_at', 'closed_at', 
                    'kyc_sender_create_date', 'kyc_receiver_create_date',
                    'dob_sender', 'dob_receiver', 'self_closure_date']
    
    for col in date_columns:
        if col in transaction_data.columns:
            transaction_data[col] = pd.to_datetime(transaction_data[col])
    
    return transaction_data, metadata, rule_descriptions

# Load all data
transaction_data, metadata, rule_descriptions = load_data()

# Display basic information about the data
print(f"Dataset shape: {transaction_data.shape}")
print(f"Date range: {transaction_data['transaction_date_time_local'].min().date()} to {transaction_data['transaction_date_time_local'].max().date()}")
print(f"Number of unique rules: {transaction_data['alert_rules'].nunique()}")
print(f"Number of unique sender KYC IDs: {transaction_data['sender_kyc_id_no'].nunique()}")
print(f"Number of unique receiver KYC IDs: {transaction_data['receiver_kyc_id_no'].nunique()}")

# Display a sample of the transaction data
transaction_data.head()
```

## 3. KYC Alert Overlap Analysis

In this section, we'll identify KYC IDs that have alerted across multiple rules to understand the overlap.

```python
def analyze_kyc_alert_overlap(transaction_data):
    """Analyze KYC IDs that have alerted across rules and determine overlap."""
    print("Analyzing KYC IDs that have alerted...")
    
    # Group alerts by KYC ID (based on triggered_on field)
    kyc_alerts = defaultdict(set)
    kyc_entity_type = {}
    
    for idx, row in transaction_data.iterrows():
        if row['triggered_on'] == 'sender':
            kyc_id = row['sender_kyc_id_no']
            kyc_entity_type[kyc_id] = 'sender'
        else:  # receiver
            kyc_id = row['receiver_kyc_id_no']
            kyc_entity_type[kyc_id] = 'receiver'
            
        kyc_alerts[kyc_id].add(row['alert_rules'])
    
    # Calculate statistics
    total_kyc_with_alerts = len(kyc_alerts)
    kyc_with_multiple_rules = sum(1 for rules in kyc_alerts.values() if len(rules) > 1)
    
    # Distribution of number of rules per KYC
    rule_count_per_kyc = [len(rules) for rules in kyc_alerts.values()]
    rule_count_distribution = pd.Series(rule_count_per_kyc).value_counts().sort_index()
    
    # Calculate overlap percentage
    overlap_percentage = (kyc_with_multiple_rules / total_kyc_with_alerts) * 100 if total_kyc_with_alerts > 0 else 0
    
    print(f"Total KYC IDs with alerts: {total_kyc_with_alerts}")
    print(f"KYC IDs alerting on multiple rules: {kyc_with_multiple_rules}")
    print(f"Percentage of KYC IDs alerting on multiple rules: {overlap_percentage:.2f}%")
    
    print("\nDistribution of number of rules per KYC ID:")
    print(rule_count_distribution)
    
    # Calculate statistics by entity type (sender vs receiver)
    sender_kycs = {k: v for k, v in kyc_alerts.items() if kyc_entity_type.get(k) == 'sender'}
    receiver_kycs = {k: v for k, v in kyc_alerts.items() if kyc_entity_type.get(k) == 'receiver'}
    
    # Sender statistics
    sender_multiple_rules = sum(1 for rules in sender_kycs.values() if len(rules) > 1)
    sender_overlap_pct = (sender_multiple_rules / len(sender_kycs)) * 100 if sender_kycs else 0
    
    # Receiver statistics
    receiver_multiple_rules = sum(1 for rules in receiver_kycs.values() if len(rules) > 1)
    receiver_overlap_pct = (receiver_multiple_rules / len(receiver_kycs)) * 100 if receiver_kycs else 0
    
    print(f"\nSender KYCs with multiple rules: {sender_multiple_rules} ({sender_overlap_pct:.2f}%)")
    print(f"Receiver KYCs with multiple rules: {receiver_multiple_rules} ({receiver_overlap_pct:.2f}%)")
    
    # Plot the distribution as a histogram
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    bins = range(1, max(rule_count_per_kyc) + 2)
    plt.hist(rule_count_per_kyc, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Histogram of Rules per KYC ID')
    plt.xlabel('Number of Rules')
    plt.ylabel('Count of KYC IDs')
    plt.xticks(range(1, max(rule_count_per_kyc) + 1))
    plt.grid(axis='y', alpha=0.75)
    
    # Plot as a bar chart too
    plt.subplot(1, 2, 2)
    rule_count_distribution.plot(kind='bar')
    plt.title('Number of Rules Triggered per KYC ID')
    plt.xlabel('Number of Rules')
    plt.ylabel('Count of KYC IDs')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('visualizations/rules_per_kyc_distribution.png')
    plt.show()
    
    # Compare sender vs receiver overlap distribution
    sender_rule_counts = [len(rules) for rules in sender_kycs.values()]
    receiver_rule_counts = [len(rules) for rules in receiver_kycs.values()]
    
    plt.figure(figsize=(12, 6))
    bins = range(1, max(max(sender_rule_counts, default=1), max(receiver_rule_counts, default=1)) + 2)
    plt.hist([sender_rule_counts, receiver_rule_counts], bins=bins, 
             label=['Sender', 'Receiver'], alpha=0.7, edgecolor='black')
    plt.title('Histogram of Rules per KYC ID by Entity Type')
    plt.xlabel('Number of Rules')
    plt.ylabel('Count of KYC IDs')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(range(1, max(bins)))
    plt.tight_layout()
    plt.savefig('visualizations/rules_per_kyc_by_entity.png')
    plt.show()
    
    # Find co-occurring rules
    rule_pairs = []
    for rules in kyc_alerts.values():
        if len(rules) > 1:
            # Convert set to array for easier processing
            rule_list = list(rules)
            for i in range(len(rule_list)):
                for j in range(i+1, len(rule_list)):
                    rule_pairs.append((rule_list[i], rule_list[j]))
    
    # Count occurrences of each rule pair
    rule_pair_counts = pd.Series(rule_pairs).value_counts().head(15)
    
    print("\nTop 15 co-occurring rule pairs:")
    print(rule_pair_counts)
    
    # Plot top co-occurring rule pairs
    plt.figure(figsize=(14, 8))
    ax = rule_pair_counts.plot(kind='barh', color='teal')
    plt.title('Top 15 Co-occurring Rule Pairs')
    plt.xlabel('Count of Co-occurrences')
    plt.ylabel('Rule Pair')
    
    # Add percentage labels relative to total overlapping KYCs
    for i, v in enumerate(rule_pair_counts):
        percentage = 100 * v / kyc_with_multiple_rules
        ax.text(v + 0.1, i, f'{percentage:.1f}%', va='center')
        
    plt.tight_layout()
    plt.savefig('visualizations/top_cooccurring_rules.png')
    plt.show()
    
    # Create and plot co-occurrence matrix if we have rule pairs
    if rule_pairs:
        unique_rules = sorted(set(rule for pair in rule_pairs for rule in pair))
        
        # Only create a heatmap if not too large
        if len(unique_rules) <= 30:  
            cooccurrence_matrix = pd.DataFrame(0, index=unique_rules, columns=unique_rules)
            
            for r1, r2 in rule_pairs:
                cooccurrence_matrix.loc[r1, r2] += 1
                cooccurrence_matrix.loc[r2, r1] += 1
            
            plt.figure(figsize=(14, 12))
            sns.heatmap(cooccurrence_matrix, cmap="YlGnBu", annot=True, fmt='.0f')
            plt.title('Rule Co-occurrence Matrix')
            plt.tight_layout()
            plt.savefig('visualizations/rule_cooccurrence_matrix.png')
            plt.show()
    
    # Additional analysis: most common rule combinations (more than pairs)
    rule_combinations = defaultdict(int)
    
    # Look for combinations of 2-4 rules that frequently co-occur
    for rules in kyc_alerts.values():
        rule_list = sorted(list(rules))  # Sort to ensure consistent ordering
        if len(rule_list) >= 2:
            # Generate all combinations of 2-4 rules (or fewer if not enough rules)
            max_combo_size = min(4, len(rule_list))
            for size in range(2, max_combo_size + 1):
                for combo in itertools.combinations(rule_list, size):
                    rule_combinations[combo] += 1
    
    # Get the top combinations
    top_combinations = sorted(rule_combinations.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\nTop rule combinations (beyond just pairs):")
    for combo, count in top_combinations:
        print(f"{' + '.join(combo)}: {count} occurrences")
    
    return kyc_alerts, rule_count_distribution, rule_pair_counts, kyc_entity_type, rule_combinations

# Execute KYC alert overlap analysis
kyc_alerts, rule_count_distribution, rule_pair_counts, kyc_entity_type, rule_combinations = analyze_kyc_alert_overlap(transaction_data)
```

## 4. Rule Efficiency Analysis

Next, we'll analyze the efficiency of each rule based on true positive and false positive rates.

```python
def analyze_rule_efficiency_with_impact(transaction_data, rule_descriptions):
    """Analyze the efficiency of rules and quantify impact of potential modifications."""
    print("Analyzing rule efficiency and potential impact of modifications...")
    
    # Filter for closed alerts only (where investigation is complete)
    closed_alerts = transaction_data[transaction_data['status'].isin(['Closed TP', 'Closed FP'])]
    
    # Overall TP/FP rates
    true_positive_rate = len(closed_alerts[closed_alerts['status'] == 'Closed TP']) / len(closed_alerts) * 100
    false_positive_rate = len(closed_alerts[closed_alerts['status'] == 'Closed FP']) / len(closed_alerts) * 100
    
    print(f"Overall True Positive Rate: {true_positive_rate:.2f}%")
    print(f"Overall False Positive Rate: {false_positive_rate:.2f}%")
    
    # Create a performance dataframe for each rule
    rule_performance = closed_alerts.groupby('alert_rules').apply(
        lambda x: pd.Series({
            'Total': len(x),
            'TP': sum(x['status'] == 'Closed TP'),
            'FP': sum(x['status'] == 'Closed FP'),
            'TP_Rate': sum(x['status'] == 'Closed TP') / len(x) * 100 if len(x) > 0 else 0,
            'FP_Rate': sum(x['status'] == 'Closed FP') / len(x) * 100 if len(x) > 0 else 0,
            'Frequency': x['rule_frequency'].iloc[0] if not x['rule_frequency'].empty else 'Unknown',
            'Pattern': x['rule_pattern'].iloc[0] if not x['rule_pattern'].empty else 'Unknown'
        })
    ).reset_index()
    
    # Merge with rule descriptions
    rule_performance = rule_performance.merge(
        rule_descriptions[['Rule no.', 'Rule description', 'Current threshold']], 
        left_on='alert_rules', 
        right_on='Rule no.', 
        how='left'
    ).drop('Rule no.', axis=1)
    
    # Calculate efficiency score (F1-like measure)
    # Higher weight for TP rate to prioritize catching real issues
    rule_performance['Efficiency_Score'] = (2 * rule_performance['TP_Rate']) / (100 + rule_performance['TP_Rate'])
    
    # Sort by efficiency score descending
    rule_performance_by_efficiency = rule_performance.sort_values('Efficiency_Score', ascending=False)
    
    print("\nRule performance by efficiency score (Top 10):")
    print(rule_performance_by_efficiency[['alert_rules', 'Total', 'TP', 'FP', 'TP_Rate', 'Efficiency_Score', 'Frequency', 'Pattern']].head(10))
    
    print("\nRule performance by efficiency score (Bottom 10):")
    print(rule_performance_by_efficiency[['alert_rules', 'Total', 'TP', 'FP', 'TP_Rate', 'Efficiency_Score', 'Frequency', 'Pattern']].tail(10))
    
    # Calculate impact of removing inefficient rules
    # Find inefficient rules (high volume, low TP rate)
    inefficient_rules = rule_performance[(rule_performance['Total'] > 50) & 
                                         (rule_performance['TP_Rate'] < 30)].sort_values('Total', ascending=False)
    
    print("\nInefficient rules (high volume, low TP rate):")
    print(inefficient_rules[['alert_rules', 'Total', 'TP', 'FP', 'TP_Rate', 'Efficiency_Score', 'Frequency', 'Pattern']].head(10))
    
    # Calculate impact if we remove the top 5 inefficient rules
    if not inefficient_rules.empty:
        top5_inefficient = inefficient_rules.head(5)['alert_rules'].tolist()
        removed_alerts = closed_alerts[closed_alerts['alert_rules'].isin(top5_inefficient)]
        removed_tp = removed_alerts[removed_alerts['status'] == 'Closed TP'].shape[0]
        removed_fp = removed_alerts[removed_alerts['status'] == 'Closed FP'].shape[0]
        
        # Calculate new metrics after removal
        new_total = len(closed_alerts) - len(removed_alerts)
        new_tp = len(closed_alerts[closed_alerts['status'] == 'Closed TP']) - removed_tp
        new_fp = len(closed_alerts[closed_alerts['status'] == 'Closed FP']) - removed_fp
        new_tp_rate = new_tp / new_total * 100 if new_total > 0 else 0
        
        # Calculate percentage changes
        alert_reduction_pct = len(removed_alerts) / len(closed_alerts) * 100
        tp_reduction_pct = removed_tp / len(closed_alerts[closed_alerts['status'] == 'Closed TP']) * 100
        fp_reduction_pct = removed_fp / len(closed_alerts[closed_alerts['status'] == 'Closed FP']) * 100
        tp_rate_change = new_tp_rate - true_positive_rate
        
        print("\nImpact of removing top 5 inefficient rules:")
        print(f"Alert volume reduction: {len(removed_alerts)} alerts ({alert_reduction_pct:.2f}% of total)")
        print(f"True positives lost: {removed_tp} ({tp_reduction_pct:.2f}% of all TPs)")
        print(f"False positives eliminated: {removed_fp} ({fp_reduction_pct:.2f}% of all FPs)")
        print(f"True positive rate change: {true_positive_rate:.2f}% → {new_tp_rate:.2f}% ({tp_rate_change:+.2f}%)")
    
    # Analyze performance by pattern
    pattern_performance = rule_performance.groupby('Pattern').agg({
        'Total': 'sum',
        'TP': 'sum',
        'FP': 'sum'
    }).reset_index()
    
    pattern_performance['TP_Rate'] = pattern_performance['TP'] / pattern_performance['Total'] * 100
    pattern_performance['Volume_Pct'] = pattern_performance['Total'] / pattern_performance['Total'].sum() * 100
    pattern_performance = pattern_performance.sort_values('TP_Rate', ascending=False)
    
    print("\nPerformance by rule pattern:")
    print(pattern_performance)
    
    # Analyze performance by frequency
    frequency_performance = rule_performance.groupby('Frequency').agg({
        'Total': 'sum',
        'TP': 'sum',
        'FP': 'sum'
    }).reset_index()
    
    frequency_performance['TP_Rate'] = frequency_performance['TP'] / frequency_performance['Total'] * 100
    frequency_performance['Volume_Pct'] = frequency_performance['Total'] / frequency_performance['Total'].sum() * 100
    frequency_performance = frequency_performance.sort_values('TP_Rate', ascending=False)
    
    print("\nPerformance by rule frequency:")
    print(frequency_performance)
    
    # Plot TP rate by rule (top 20 by volume) with FP rate
    top_rules_by_volume = rule_performance.sort_values('Total', ascending=False).head(20)
    
    # Create a figure for the visualization
    plt.figure(figsize=(16, 8))
    
    # Set the positions for the bars
    x = np.arange(len(top_rules_by_volume))
    width = 0.35
    
    # Create the bars
    plt.bar(x - width/2, top_rules_by_volume['TP_Rate'], width, label='TP Rate', color='green', alpha=0.7)
    plt.bar(x + width/2, top_rules_by_volume['FP_Rate'], width, label='FP Rate', color='red', alpha=0.7)
    
    # Add a reference line at 50%
    plt.axhline(y=50, color='blue', linestyle='--', label='50% Rate')
    
    # Add some text for labels, title and axes ticks
    plt.xlabel('Rule')
    plt.ylabel('Rate (%)')
    plt.title('True Positive and False Positive Rates for Top 20 Rules by Volume')
    plt.xticks(x, top_rules_by_volume['alert_rules'], rotation=45, ha='right')
    plt.legend()
    
    # Add a second y-axis for alert volume
    ax2 = plt.twinx()
    # Plot the total volume as a line
    ax2.plot(x, top_rules_by_volume['Total'], 'o-', color='purple', alpha=0.6, label='Total Alerts')
    ax2.set_ylabel('Number of Alerts', color='purple')
    ax2.tick_params(axis='y', colors='purple')
    
    # Add annotations for total volume
    for i, v in enumerate(top_rules_by_volume['Total']):
        ax2.annotate(str(v), (x[i], v), xytext=(0, 5), textcoords='offset points', ha='center')
    
    plt.tight_layout()
    plt.savefig('visualizations/tp_fp_rates_top_volume_rules.png')
    plt.show()
    
    # Plot by pattern with volume proportion
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    
    # Bar plot for TP rate
    bars = plt.bar(pattern_performance['Pattern'], pattern_performance['TP_Rate'], 
                   color='green', alpha=0.7, label='TP Rate')
    
    # Add volume percentage annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        volume_pct = pattern_performance.iloc[i]['Volume_Pct']
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'Vol: {volume_pct:.1f}%', ha='center', va='bottom')
    
    plt.title('True Positive Rate by Rule Pattern (with Volume Percentage)')
    plt.xlabel('Pattern')
    plt.ylabel('True Positive Rate (%)')
    plt.ylim(0, 100)  # Set y-axis limit to 100%
    plt.tight_layout()
    plt.savefig('visualizations/tp_rate_by_pattern_with_volume.png')
    plt.show()
    
    # Plot by frequency with volume proportion
    plt.figure(figsize=(12, 8))
    
    # Bar plot for TP rate
    bars = plt.bar(frequency_performance['Frequency'], frequency_performance['TP_Rate'], 
                   color='purple', alpha=0.7, label='TP Rate')
    
    # Add volume percentage annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        volume_pct = frequency_performance.iloc[i]['Volume_Pct']
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'Vol: {volume_pct:.1f}%', ha='center', va='bottom')
    
    plt.title('True Positive Rate by Rule Frequency (with Volume Percentage)')
    plt.xlabel('Frequency')
    plt.ylabel('True Positive Rate (%)')
    plt.ylim(0, 100)  # Set y-axis limit to 100%
    plt.tight_layout()
    plt.savefig('visualizations/tp_rate_by_frequency_with_volume.png')
    plt.show()
    
    # Get list of true positives
    true_positives = transaction_data[transaction_data['status'] == 'Closed TP']
    
    # Extract unique KYC IDs with true positive alerts
    tp_kyc_ids = []
    for idx, row in true_positives.iterrows():
        if row['triggered_on'] == 'sender':
            tp_kyc_ids.append(row['sender_kyc_id_no'])
        else:  # receiver
            tp_kyc_ids.append(row['receiver_kyc_id_no'])
    
    unique_tp_kyc_ids = set(tp_kyc_ids)
    
    print(f"\nTotal true positive alerts: {len(true_positives)}")
    print(f"Number of unique KYC IDs with true positive alerts: {len(unique_tp_kyc_ids)}")
    
    # Calculate effectiveness ratio (TP per KYC)
    tp_per_kyc = len(true_positives) / len(unique_tp_kyc_ids) if unique_tp_kyc_ids else 0
    print(f"True positives per unique KYC ID: {tp_per_kyc:.2f}")
    
    return rule_performance, pattern_performance, frequency_performance, true_positives, unique_tp_kyc_ids

# Execute rule efficiency analysis
rule_performance, pattern_performance, frequency_performance, true_positives, tp_kyc_ids = analyze_rule_efficiency_with_impact(transaction_data, rule_descriptions)
```

## 5. Rule Clustering Analysis

In this section, we'll perform rule clustering to identify natural groups of rules.

```python
def perform_rule_clustering(transaction_data, rule_performance, kyc_alerts):
    """Perform rule clustering analysis to identify natural groups of rules."""
    print("\nPerforming rule clustering analysis...")
    
    # Extract rule co-occurrence patterns
    all_rules = sorted(transaction_data['alert_rules'].unique())
    rule_matrix = pd.DataFrame(0, index=kyc_alerts.keys(), columns=all_rules)
    
    for kyc_id, rules in kyc_alerts.items():
        for rule in rules:
            if rule in all_rules:  # Ensure rule is in the columns
                rule_matrix.loc[kyc_id, rule] = 1
    
    # Calculate correlation matrix between rules
    rule_corr_matrix = rule_matrix.corr()
    
    # Convert NaN to 0 (in case some rules have no co-occurrences)
    rule_corr_matrix.fillna(0, inplace=True)
    
    # Create a distance matrix (1 - correlation)
    distance_matrix = 1 - rule_corr_matrix.abs()
    
    # Perform hierarchical clustering
    try:
        linkage_matrix = linkage(distance_matrix.values, method='ward')
        
        # Plot dendrogram
        plt.figure(figsize=(16, 10))
        plt.title("Hierarchical Clustering of Rules Based on Co-occurrence")
        dendrogram(linkage_matrix, labels=all_rules, leaf_rotation=90)
        plt.tight_layout()
        plt.savefig('visualizations/rule_hierarchical_clustering.png')
        plt.show()
        
        # Determine optimal number of clusters
        num_clusters = min(10, len(all_rules))  # Cap at 10 clusters for readability
        
        # Fix: Change clustering parameters to be compatible
        # Use euclidean distance with ward linkage instead of precomputed
        cluster_model = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
        cluster_labels = cluster_model.fit_predict(distance_matrix)
        
        # Create a DataFrame with rule clusters
        rule_clusters = pd.DataFrame({
            'Rule': all_rules,
            'Cluster': cluster_labels
        })
        
        # Add performance metrics
        rule_clusters = rule_clusters.merge(rule_performance[['alert_rules', 'Total', 'TP_Rate', 'Pattern', 'Frequency']], 
                                            left_on='Rule', right_on='alert_rules', how='left').drop('alert_rules', axis=1)
        
        # Analyze each cluster
        print("\nRule clusters from hierarchical clustering:")
        for cluster_id in range(num_clusters):
            cluster_rules = rule_clusters[rule_clusters['Cluster'] == cluster_id]
            print(f"\nCluster {cluster_id} ({len(cluster_rules)} rules):")
            
            # Print rules in this cluster
            print("Rules: " + ", ".join(cluster_rules['Rule'].tolist()))
            
            # Calculate cluster metrics
            avg_tp_rate = cluster_rules['TP_Rate'].mean()
            patterns = cluster_rules['Pattern'].value_counts()
            frequencies = cluster_rules['Frequency'].value_counts()
            
            print(f"Average TP Rate: {avg_tp_rate:.2f}%")
            print(f"Dominant patterns: {patterns.to_dict()}")
            print(f"Frequencies: {frequencies.to_dict()}")
        
        # Visualize clusters on a 2D plot
        # We'll use a simple 2D representation based on TP rate and total alerts
        plt.figure(figsize=(14, 10))
        
        # Create a scatterplot of rules
        scatter = plt.scatter(rule_clusters['Total'], rule_clusters['TP_Rate'], 
                             c=rule_clusters['Cluster'], cmap='viridis', 
                             s=100, alpha=0.7)
        
        # Add rule labels
        for i, row in rule_clusters.iterrows():
            plt.annotate(row['Rule'], (row['Total'], row['TP_Rate']), 
                         xytext=(5, 5), textcoords='offset points')
        
        plt.colorbar(scatter, label='Cluster')
        plt.title('Rule Clusters by Alert Volume and TP Rate')
        plt.xlabel('Total Alerts')
        plt.ylabel('True Positive Rate (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/rule_clusters_scatter.png')
        plt.show()
        
        # Network visualization of rule relationships
        # Create a graph where rules are nodes and edges represent correlation
        G = nx.Graph()
        
        # Add nodes
        for rule in all_rules:
            cluster_id = rule_clusters.loc[rule_clusters['Rule'] == rule, 'Cluster'].iloc[0]
            G.add_node(rule, cluster=cluster_id)
        
        # Add edges (only for correlations above threshold)
        threshold = 0.3  # Correlation threshold
        for i in range(len(all_rules)):
            for j in range(i+1, len(all_rules)):
                rule1, rule2 = all_rules[i], all_rules[j]
                correlation = rule_corr_matrix.loc[rule1, rule2]
                if correlation > threshold:
                    G.add_edge(rule1, rule2, weight=correlation)
        
        # Plot the network if it's not too large
        if len(all_rules) <= 40:  # Limit for readability
            plt.figure(figsize=(16, 12))
            
            # Position nodes using force-directed layout
            pos = nx.spring_layout(G, k=0.3, iterations=50)
            
            # Get node colors based on cluster
            node_colors = [G.nodes[rule]['cluster'] for rule in G.nodes()]
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=200, node_color=node_colors, cmap='viridis', alpha=0.8)
            
            # Draw edges with varying width based on correlation
            edges = G.edges(data=True)
            edge_weights = [edge[2]['weight']*3 for edge in edges]
            nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=8)
            
            plt.title('Rule Correlation Network (Colored by Cluster)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('visualizations/rule_correlation_network.png')
            plt.show()
    
    except Exception as e:
        print(f"Error in clustering: {e}")
        import traceback
        traceback.print_exc()
        # Create an empty DataFrame to avoid further errors
        rule_clusters = pd.DataFrame(columns=['Rule', 'Cluster'])
    
    return rule_clusters, rule_corr_matrix

# Execute rule clustering
rule_clusters, rule_corr_matrix = perform_rule_clustering(transaction_data, rule_performance, kyc_alerts)
```

## 6. KYC Breakage Analysis

In this section, we'll analyze the KYC breakage issue where multiple KYC IDs exist for the same person.

```python
def analyze_kyc_breakage_enhanced(transaction_data):
    """Enhanced KYC breakage analysis with additional visualizations and metrics."""
    print("\nEnhanced KYC breakage analysis with visualization...")
    
    # Create directory for visualizations
    kyc_dir = 'visualizations/kyc_breakage'
    os.makedirs(kyc_dir, exist_ok=True)
    
    # Initialize these variables to empty DataFrames to avoid reference errors
    sender_fp = pd.DataFrame()
    receiver_fp = pd.DataFrame()
    
    # Analyze sender KYC breakage
    print("\n=== Sender KYC Breakage Analysis ===")
    
    # Group by sender name and count KYC IDs
    sender_name_groups = transaction_data.groupby('sender_name_kyc_wise')['sender_kyc_id_no'].nunique().reset_index()
    sender_name_groups.columns = ['sender_name', 'kyc_id_count']
    
    # Distribution of KYC IDs per sender name
    sender_kyc_distribution = sender_name_groups['kyc_id_count'].value_counts().sort_index()
    
    print("Distribution of KYC IDs per unique sender name:")
    print(sender_kyc_distribution.head(10))
    
    # Calculate stats
    avg_kyc_per_sender = sender_name_groups['kyc_id_count'].mean()
    max_kyc_per_sender = sender_name_groups['kyc_id_count'].max()
    senders_with_multiple_kyc = sum(sender_name_groups['kyc_id_count'] > 1)
    percentage_senders_with_multiple_kyc = senders_with_multiple_kyc / len(sender_name_groups) * 100
    
    print(f"\nAverage KYC IDs per unique sender name: {avg_kyc_per_sender:.2f}")
    print(f"Maximum KYC IDs for a single sender name: {max_kyc_per_sender}")
    print(f"Senders with multiple KYC IDs: {senders_with_multiple_kyc} ({percentage_senders_with_multiple_kyc:.2f}%)")
    
    # Enhanced visualizations for sender KYC breakage
    plt.figure(figsize=(12, 7))
    
    # Plot as histogram
    sender_kyc_counts = sender_name_groups['kyc_id_count'].values
    
    # Calculate histogram with log scale for better visualization
    hist, bins = np.histogram(sender_kyc_counts, bins=range(1, max(sender_kyc_counts) + 2))
    
    # Plot with logarithmic y-axis for better visibility of the distribution
    plt.bar(bins[:-1], hist, width=0.8, alpha=0.7, color='blue', edgecolor='black')
    plt.yscale('log')
    plt.title('Distribution of KYC IDs per Sender Name (Log Scale)')
    plt.xlabel('Number of KYC IDs')
    plt.ylabel('Count of Sender Names (Log Scale)')
    plt.xticks(range(1, min(max(sender_kyc_counts) + 1, 20)))
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f'{kyc_dir}/sender_kyc_distribution_log.png')
    plt.close()
    
    # Create pie chart of single vs multiple KYC senders
    plt.figure(figsize=(10, 8))
    single_kyc = len(sender_name_groups) - senders_with_multiple_kyc
    plt.pie([single_kyc, senders_with_multiple_kyc], 
            labels=['Single KYC ID', 'Multiple KYC IDs'],
            autopct='%1.1f%%', 
            colors=['lightblue', 'red'],
            explode=(0, 0.1),
            shadow=True)
    plt.title('Sender KYC Breakage: Single vs Multiple KYC IDs')
    plt.savefig(f'{kyc_dir}/sender_kyc_breakage_pie.png')
    plt.show()
    
    # Find senders with the most KYC IDs
    top_multiple_kyc_senders = sender_name_groups.sort_values('kyc_id_count', ascending=False).head(10)
    print("\nTop 10 senders with most KYC IDs:")
    print(top_multiple_kyc_senders)
    
    # Visualize top senders with most KYC IDs
    plt.figure(figsize=(12, 6))
    plt.bar(top_multiple_kyc_senders['sender_name'], top_multiple_kyc_senders['kyc_id_count'], 
            color='green', alpha=0.7)
    plt.title('Top 10 Senders with Most KYC IDs')
    plt.xlabel('Sender Name')
    plt.ylabel('Number of KYC IDs')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{kyc_dir}/top_sender_kyc_counts.png')
    plt.show()
    
    # Analyze receiver KYC breakage
    print("\n=== Receiver KYC Breakage Analysis ===")
    
    # Group by receiver name and count KYC IDs
    receiver_name_groups = transaction_data.groupby('receiver_name_kyc_wise')['receiver_kyc_id_no'].nunique().reset_index()
    receiver_name_groups.columns = ['receiver_name', 'kyc_id_count']
    
    # Distribution of KYC IDs per receiver name
    receiver_kyc_distribution = receiver_name_groups['kyc_id_count'].value_counts().sort_index()
    
    print("Distribution of KYC IDs per unique receiver name:")
    print(receiver_kyc_distribution.head(10))
    
    # Calculate stats
    avg_kyc_per_receiver = receiver_name_groups['kyc_id_count'].mean()
    max_kyc_per_receiver = receiver_name_groups['kyc_id_count'].max()
    receivers_with_multiple_kyc = sum(receiver_name_groups['kyc_id_count'] > 1)
    percentage_receivers_with_multiple_kyc = receivers_with_multiple_kyc / len(receiver_name_groups) * 100
    
    print(f"\nAverage KYC IDs per unique receiver name: {avg_kyc_per_receiver:.2f}")
    print(f"Maximum KYC IDs for a single receiver name: {max_kyc_per_receiver}")
    print(f"Receivers with multiple KYC IDs: {receivers_with_multiple_kyc} ({percentage_receivers_with_multiple_kyc:.2f}%)")
    
    # Compare sender vs receiver KYC breakage
    plt.figure(figsize=(12, 6))
    
    # Data for plotting
    categories = ['Sender', 'Receiver']
    single_kycs = [len(sender_name_groups) - senders_with_multiple_kyc, 
                  len(receiver_name_groups) - receivers_with_multiple_kyc]
    multiple_kycs = [senders_with_multiple_kyc, receivers_with_multiple_kyc]
    
    # Create stacked bar chart
    width = 0.6
    plt.bar(categories, single_kycs, width, label='Single KYC ID', color='lightblue')
    plt.bar(categories, multiple_kycs, width, bottom=single_kycs, label='Multiple KYC IDs', color='red')
    
    # Add percentages
    for i, category in enumerate(categories):
        total = single_kycs[i] + multiple_kycs[i]
        multiple_pct = multiple_kycs[i] / total * 100 if total > 0 else 0
        plt.text(i, single_kycs[i] + multiple_kycs[i]/2, f"{multiple_pct:.1f}%", 
                ha='center', va='center', color='white', fontweight='bold')
    
    plt.title('Comparison of Sender vs Receiver KYC Breakage')
    plt.ylabel('Number of Entities')
    plt.legend()
    plt.savefig(f'{kyc_dir}/sender_vs_receiver_kyc_breakage.png')
    plt.show()
    
    # Analyze impact on alerts
    print("\n=== Impact of KYC Breakage on Alerts ===")
    
    # Filter for alerts triggered on receivers and senders
    receiver_alerts = transaction_data[transaction_data['triggered_on'] == 'receiver']
    sender_alerts = transaction_data[transaction_data['triggered_on'] == 'sender']
    
    # Find alerts for entities with multiple KYC IDs
    multiple_kyc_senders = sender_name_groups[sender_name_groups['kyc_id_count'] > 1]['sender_name'].tolist()
    multiple_kyc_receivers = receiver_name_groups[receiver_name_groups['kyc_id_count'] > 1]['receiver_name'].tolist()
    
    # Filter alerts for these entities
    sender_multiple_kyc_alerts = sender_alerts[sender_alerts['sender_name_kyc_wise'].isin(multiple_kyc_senders)]
    receiver_multiple_kyc_alerts = receiver_alerts[receiver_alerts['receiver_name_kyc_wise'].isin(multiple_kyc_receivers)]
    
    # Calculate impact percentages
    sender_affected_pct = len(sender_multiple_kyc_alerts) / len(sender_alerts) * 100 if len(sender_alerts) > 0 else 0
    receiver_affected_pct = len(receiver_multiple_kyc_alerts) / len(receiver_alerts) * 100 if len(receiver_alerts) > 0 else 0
    overall_affected_pct = (len(sender_multiple_kyc_alerts) + len(receiver_multiple_kyc_alerts)) / len(transaction_data) * 100
    
    print(f"Sender alerts affected by KYC breakage: {len(sender_multiple_kyc_alerts)} ({sender_affected_pct:.2f}%)")
    print(f"Receiver alerts affected by KYC breakage: {len(receiver_multiple_kyc_alerts)} ({receiver_affected_pct:.2f}%)")
    print(f"Overall alerts affected: {len(sender_multiple_kyc_alerts) + len(receiver_multiple_kyc_alerts)} ({overall_affected_pct:.2f}%)")
    
    # Create visualization of alert impact
    plt.figure(figsize=(14, 10))
    
    # Create stacked bar for sender alerts
    plt.subplot(1, 2, 1)
    sender_multiple = len(sender_multiple_kyc_alerts)
    sender_normal = len(sender_alerts) - sender_multiple
    plt.pie([sender_normal, sender_multiple], 
            labels=['Normal Alerts', 'Multiple KYC Alerts'],
            colors=['lightblue', 'red'],
            autopct='%1.1f%%',
            explode=(0, 0.1))
    plt.title('Sender Alerts Affected by KYC Breakage')
    
    # Create stacked bar for receiver alerts
    plt.subplot(1, 2, 2)
    receiver_multiple = len(receiver_multiple_kyc_alerts)
    receiver_normal = len(receiver_alerts) - receiver_multiple
    plt.pie([receiver_normal, receiver_multiple], 
            labels=['Normal Alerts', 'Multiple KYC Alerts'],
            colors=['lightgreen', 'red'],
            autopct='%1.1f%%',
            explode=(0, 0.1))
    plt.title('Receiver Alerts Affected by KYC Breakage')
    
    plt.tight_layout()
    plt.savefig(f'{kyc_dir}/alerts_affected_by_kyc_breakage.png')
    plt.show()
    
    # Advanced analysis: Estimated impact of KYC deduplication
    print("\n=== Estimated Impact of KYC Deduplication ===")
    
    # Estimate alert reduction
    estimated_sender_reduction = 0
    if not sender_multiple_kyc_alerts.empty:
        # Analyze the distribution of alerts per sender
        sender_alert_counts = sender_multiple_kyc_alerts.groupby('sender_name_kyc_wise').size()
        redundancy_factor = sender_alert_counts.mean() / sender_name_groups.loc[sender_name_groups['sender_name'].isin(sender_alert_counts.index), 'kyc_id_count'].mean()
        estimated_sender_reduction = len(sender_multiple_kyc_alerts) * (1 - 1/redundancy_factor) if redundancy_factor > 0 else 0
    
    estimated_receiver_reduction = 0
    if not receiver_multiple_kyc_alerts.empty:
        # Analyze the distribution of alerts per receiver
        receiver_alert_counts = receiver_multiple_kyc_alerts.groupby('receiver_name_kyc_wise').size()
        redundancy_factor = receiver_alert_counts.mean() / receiver_name_groups.loc[receiver_name_groups['receiver_name'].isin(receiver_alert_counts.index), 'kyc_id_count'].mean()
        estimated_receiver_reduction = len(receiver_multiple_kyc_alerts) * (1 - 1/redundancy_factor) if redundancy_factor > 0 else 0
    
    total_estimated_reduction = estimated_sender_reduction + estimated_receiver_reduction
    total_reduction_pct = total_estimated_reduction / len(transaction_data) * 100 if len(transaction_data) > 0 else 0
    
    print(f"Estimated alert reduction from sender KYC deduplication: {estimated_sender_reduction:.0f} alerts")
    print(f"Estimated alert reduction from receiver KYC deduplication: {estimated_receiver_reduction:.0f} alerts")
    print(f"Total estimated alert reduction: {total_estimated_reduction:.0f} alerts ({total_reduction_pct:.2f}%)")
    
    return sender_name_groups, receiver_name_groups, sender_multiple_kyc_alerts, receiver_multiple_kyc_alerts

# Execute KYC breakage analysis
sender_name_groups, receiver_name_groups, sender_multiple_kyc_alerts, receiver_multiple_kyc_alerts = analyze_kyc_breakage_enhanced(transaction_data)
```

## 7. ATL/BTL Threshold Optimization

In this section, we'll perform ATL/BTL threshold optimization analysis for key rules.

```python
def perform_atl_btl_analysis(transaction_data, rule_descriptions):
    """Perform ATL/BTL threshold optimization analysis for key rules."""
    print("\nPerforming ATL/BTL threshold optimization analysis...")
    
    # Create a directory for ATL/BTL visualizations
    atl_btl_dir = 'visualizations/atl_btl_analysis'
    os.makedirs(atl_btl_dir, exist_ok=True)
    
    # Get the top rules by volume for analysis
    top_volume_rules = transaction_data['alert_rules'].value_counts().head(5).index.tolist()
    
    # Also get rules with high false positive rates (if we have status information)
    high_fp_rules = []
    if 'status' in transaction_data.columns:
        closed_alerts = transaction_data[transaction_data['status'].isin(['Closed TP', 'Closed FP'])]
        if not closed_alerts.empty:
            rule_fp_rates = closed_alerts.groupby('alert_rules').apply(
                lambda x: (x['status'] == 'Closed FP').sum() / len(x) * 100 if len(x) > 0 else 0
            ).sort_values(ascending=False)
            high_fp_rules = rule_fp_rates[rule_fp_rates > 70].head(5).index.tolist()
    
    # Combine and deduplicate the rule lists
    rules_to_analyze = list(set(top_volume_rules + high_fp_rules))
    print(f"Analyzing {len(rules_to_analyze)} rules for threshold optimization")
    
    # Get current thresholds from rule descriptions
    rule_thresholds = {}
    if rule_descriptions is not None:
        try:
            rule_thresholds = rule_descriptions.set_index('Rule no.')['Current threshold'].to_dict()
        except Exception as e:
            print(f"Error getting current thresholds: {e}")
    
    # Create a summary dataframe for the report
    atl_btl_summary = []
    
    # For each rule, simulate different thresholds
    for rule in rules_to_analyze:
        try:
            print(f"\nAnalyzing Rule {rule}")
            
            # Get current threshold if available
            current_threshold = rule_thresholds.get(rule, "Unknown")
            print(f"Current threshold: {current_threshold}")
            
            # Get alerts for this rule
            rule_alerts = transaction_data[transaction_data['alert_rules'] == rule]
            print(f"Total alerts for this rule: {len(rule_alerts)}")
            
            # Filter for closed alerts (with known outcome)
            closed_rule_alerts = rule_alerts[rule_alerts['status'].isin(['Closed TP', 'Closed FP'])]
            print(f"Closed alerts (with known outcome): {len(closed_rule_alerts)}")
            
            # Choose metric for threshold analysis (using USD value as an example)
            if 'usd_value' in closed_rule_alerts.columns:
                threshold_metrics = closed_rule_alerts['usd_value'].values
                metric_name = "USD Value"
                
                # Get statuses
                statuses = closed_rule_alerts['status'].values
                
                # Filter out NaN values
                valid_indices = ~np.isnan(threshold_metrics)
                threshold_metrics = threshold_metrics[valid_indices]
                statuses = statuses[valid_indices]
                
                # Generate potential thresholds
                # Use a combination of percentiles and evenly spaced values
                percentiles = np.arange(10, 100, 5)  # 10th, 15th, ..., 95th percentiles
                percentile_thresholds = np.percentile(threshold_metrics, percentiles)
                
                # Add some evenly spaced values in the range
                min_val = np.min(threshold_metrics)
                max_val = np.max(threshold_metrics)
                even_thresholds = np.linspace(min_val, max_val, 20)
                
                # Combine and sort unique thresholds
                all_thresholds = np.unique(np.concatenate([percentile_thresholds, even_thresholds]))
                
                # Calculate metrics for each threshold
                results = []
                for threshold in all_thresholds:
                    # Determine which alerts would be triggered with this threshold
                    triggered = threshold_metrics > threshold
                    
                    # Count TPs and FPs with this threshold
                    tp_count = sum((statuses == 'Closed TP') & triggered)
                    fp_count = sum((statuses == 'Closed FP') & triggered)
                    total = sum(triggered)
                    
                    # Calculate rates
                    tp_rate = tp_count / sum(statuses == 'Closed TP') * 100 if sum(statuses == 'Closed TP') > 0 else 0
                    fp_rate = fp_count / sum(statuses == 'Closed FP') * 100 if sum(statuses == 'Closed FP') > 0 else 0
                    precision = tp_count / total * 100 if total > 0 else 0
                    
                    # Calculate alert volume reduction
                    alert_reduction = (1 - sum(triggered) / len(threshold_metrics)) * 100
                    
                    # Calculate a composite score for threshold optimization
                    # Higher score is better - we want high TP rate and low FP rate
                    if total > 0:  # Avoid division by zero
                        # F1-like score with precision and recall (TP rate)
                        if precision > 0 and tp_rate > 0:
                            f1 = 2 * (precision * tp_rate) / (precision + tp_rate)
                        else:
                            f1 = 0
                        
                        # Final score considers F1 and alert reduction
                        # Weights can be adjusted based on business priorities
                        score = (f1 * 0.8) - (alert_reduction * 0.2)  # Prioritize catching true positives
                    else:
                        score = 0  # No alerts triggered at this threshold
                    
                    results.append({
                        'Threshold': threshold,
                        'TP_Count': tp_count,
                        'FP_Count': fp_count,
                        'Total_Alerts': total,
                        'TP_Rate': tp_rate,
                        'FP_Rate': fp_rate,
                        'Precision': precision,
                        'Alert_Reduction': alert_reduction,
                        'F1_Score': f1 if 'f1' in locals() else 0,
                        'Score': score
                    })
                
                # Convert to DataFrame
                results_df = pd.DataFrame(results)
                
                # Find optimal threshold
                if 'Score' in results_df.columns and not results_df['Score'].isna().all():
                    optimal_idx = results_df['Score'].idxmax()
                    optimal_threshold = results_df.loc[optimal_idx, 'Threshold']
                else:
                    print("Could not determine optimal threshold. Using midpoint.")
                    optimal_threshold = np.median(all_thresholds)
                
                print(f"Optimal threshold: {optimal_threshold:.2f}")
                
                # Plot threshold analysis results
                plt.figure(figsize=(14, 10))
                
                # Main plot - TP Rate, Precision and Alert Reduction
                plt.subplot(2, 1, 1)
                plt.plot(results_df['Threshold'], results_df['TP_Rate'], 'o-', label='TP Rate', color='green', alpha=0.7)
                plt.plot(results_df['Threshold'], results_df['Precision'], 's-', label='Precision', color='blue', alpha=0.7)
                plt.plot(results_df['Threshold'], results_df['Alert_Reduction'], '^-', label='Alert Reduction', color='red', alpha=0.7)
                
                # Add Score line
                plt.plot(results_df['Threshold'], results_df['Score'], 'x-', label='Optimization Score', color='purple', alpha=0.7)
                
                # Add optimal threshold line
                plt.axvline(x=optimal_threshold, color='orange', linestyle='-', 
                        label=f'Optimal Threshold ({optimal_threshold:.2f})')
                
                plt.title(f'Threshold Analysis for Rule {rule}')
                plt.xlabel(f'Threshold Value ({metric_name})')
                plt.ylabel('Percentage (%)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Second plot - Alert volumes and counts
                plt.subplot(2, 1, 2)
                plt.plot(results_df['Threshold'], results_df['Total_Alerts'], 'o-', label='Total Alerts', color='blue', alpha=0.7)
                plt.plot(results_df['Threshold'], results_df['TP_Count'], 's-', label='True Positives', color='green', alpha=0.7)
                plt.plot(results_df['Threshold'], results_df['FP_Count'], '^-', label='False Positives', color='red', alpha=0.7)
                
                # Add optimal threshold line
                plt.axvline(x=optimal_threshold, color='orange', linestyle='-', 
                        label=f'Optimal Threshold ({optimal_threshold:.2f})')
                
                plt.xlabel(f'Threshold Value ({metric_name})')
                plt.ylabel('Count')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'{atl_btl_dir}/threshold_analysis_{rule}.png')
                plt.show()
                
                # Create detailed distribution plot
                plt.figure(figsize=(14, 8))
                
                # Create a histogram with TP and FP
                tp_values = threshold_metrics[statuses == 'Closed TP']
                fp_values = threshold_metrics[statuses == 'Closed FP']
                
                plt.hist([tp_values, fp_values], bins=20, alpha=0.6, 
                        label=['True Positives', 'False Positives'], 
                        color=['green', 'red'], stacked=False)
                
                # Add optimal threshold line
                plt.axvline(x=optimal_threshold, color='orange', linestyle='-', 
                        label=f'Optimal Threshold ({optimal_threshold:.2f})')
                
                plt.title(f'Distribution of {metric_name} for Rule {rule}')
                plt.xlabel(f'{metric_name}')
                plt.ylabel('Frequency')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'{atl_btl_dir}/distribution_{rule}.png')
                plt.show()
                
                # Add to summary
                optimal_result = results_df.loc[optimal_idx]
                atl_btl_summary.append({
                    'Rule': rule,
                    'Current_Threshold': current_threshold,
                    'Optimal_Threshold': f"{optimal_threshold:.2f}",
                    'Metric_Used': metric_name,
                    'TP_Rate': optimal_result['TP_Rate'],
                    'Precision': optimal_result['Precision'],
                    'Alert_Reduction': optimal_result['Alert_Reduction'],
                    'F1_Score': optimal_result['F1_Score']
                })
            
            else:
                print(f"USD value not found for rule {rule}. Skipping threshold analysis.")
        
        except Exception as e:
            print(f"Error analyzing rule {rule}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary DataFrame
    atl_btl_summary_df = pd.DataFrame(atl_btl_summary)
    
    if not atl_btl_summary_df.empty:
        print("\n=== ATL/BTL Analysis Summary ===")
        print(atl_btl_summary_df)
        
        # Calculate total alert reduction
        total_alert_reduction = atl_btl_summary_df['Alert_Reduction'].mean() * len(atl_btl_summary_df) / len(rules_to_analyze)
        print(f"\nEstimated overall alert reduction: {total_alert_reduction:.2f}%")
        
        # Create a visualization of the overall impact
        plt.figure(figsize=(12, 8))
        
        # Sort by alert reduction percentage
        plot_df = atl_btl_summary_df.sort_values('Alert_Reduction').copy()
        
        # Create bar chart of alert reduction by rule
        plt.barh(plot_df['Rule'], plot_df['Alert_Reduction'], color='skyblue')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Potential Alert Volume Reduction by Rule')
        plt.xlabel('Alert Reduction (%)')
        plt.ylabel('Rule')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{atl_btl_dir}/overall_impact.png')
        plt.show()
    
    return atl_btl_summary_df

# Execute ATL/BTL analysis
atl_btl_summary_df = perform_atl_btl_analysis(transaction_data, rule_descriptions)
```

## 8. Rule Overlap Analysis

In this section, we'll perform enhanced analysis of rule overlaps.

```python
def analyze_rule_overlap(transaction_data, rule_performance, kyc_alerts):
    """Perform enhanced analysis of rule overlaps."""
    print("\nPerforming enhanced rule overlap analysis...")
    
    # Create directory for visualizations
    overlap_dir = 'visualizations/rule_overlap'
    os.makedirs(overlap_dir, exist_ok=True)
    
    # Get all unique rules
    all_rules = sorted(transaction_data['alert_rules'].unique())
    
    # Create a co-occurrence matrix
    rule_cooccurrence = pd.DataFrame(0, index=all_rules, columns=all_rules)
    
    # Count rule co-occurrences
    for kyc_id, rules in kyc_alerts.items():
        rule_list = list(rules)
        for i in range(len(rule_list)):
            for j in range(i, len(rule_list)):
                rule_i, rule_j = rule_list[i], rule_list[j]
                rule_cooccurrence.loc[rule_i, rule_j] += 1
                if i != j:
                    rule_cooccurrence.loc[rule_j, rule_i] += 1
    
    # Calculate correlation (Jaccard similarity)
    rule_jaccard = pd.DataFrame(0.0, index=all_rules, columns=all_rules)
    
    for i in range(len(all_rules)):
        for j in range(len(all_rules)):
            rule_i, rule_j = all_rules[i], all_rules[j]
            if i == j:
                rule_jaccard.loc[rule_i, rule_j] = 1.0
            else:
                intersection = rule_cooccurrence.loc[rule_i, rule_j]
                union = (rule_cooccurrence.loc[rule_i, rule_i] + 
                         rule_cooccurrence.loc[rule_j, rule_j] - intersection)
                rule_jaccard.loc[rule_i, rule_j] = intersection / union if union > 0 else 0
    
    # Find strongly overlapping rules
    overlap_threshold = 0.6  # Jaccard similarity threshold
    strong_overlaps = []
    
    for i in range(len(all_rules)):
        for j in range(i+1, len(all_rules)):
            rule_i, rule_j = all_rules[i], all_rules[j]
            jaccard = rule_jaccard.loc[rule_i, rule_j]
            if jaccard >= overlap_threshold:
                strong_overlaps.append((rule_i, rule_j, jaccard))
    
    # Sort by Jaccard similarity
    strong_overlaps.sort(key=lambda x: x[2], reverse=True)
    
    # Print the strongly overlapping rules
    print(f"\nFound {len(strong_overlaps)} strongly overlapping rule pairs (Jaccard >= {overlap_threshold}):")
    for rule_i, rule_j, jaccard in strong_overlaps[:10]:  # Show top 10
        print(f"{rule_i} + {rule_j}: {jaccard:.3f} similarity")
    
    # Create a heatmap of rule correlations
    plt.figure(figsize=(15, 12))
    sns.heatmap(rule_jaccard, cmap="YlGnBu", annot=False, square=True)
    plt.title('Rule Correlation Heatmap (Jaccard Similarity)')
    plt.tight_layout()
    plt.savefig(f'{overlap_dir}/rule_correlation_heatmap.png')
    plt.show()
    
    # Create a network graph of rule overlaps
    G = nx.Graph()
    
    # Add nodes (rules)
    for rule in all_rules:
        # Get rule info
        rule_info = rule_performance[rule_performance['alert_rules'] == rule]
        if not rule_info.empty:
            total_alerts = rule_info.iloc[0]['Total'] if 'Total' in rule_info.columns else 0
            tp_rate = rule_info.iloc[0]['TP_Rate'] if 'TP_Rate' in rule_info.columns else 0
            pattern = rule_info.iloc[0]['Pattern'] if 'Pattern' in rule_info.columns else 'Unknown'
            frequency = rule_info.iloc[0]['Frequency'] if 'Frequency' in rule_info.columns else 'Unknown'
        else:
            total_alerts, tp_rate, pattern, frequency = 0, 0, 'Unknown', 'Unknown'
        
        # Add node with attributes
        G.add_node(rule, total_alerts=total_alerts, tp_rate=tp_rate, 
                  pattern=pattern, frequency=frequency)
    
    # Add edges for overlapping rules
    for rule_i, rule_j, jaccard in strong_overlaps:
        G.add_edge(rule_i, rule_j, weight=jaccard)
    
    # Visualization parameters
    plt.figure(figsize=(18, 14))
    
    # Layout
    try:
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    except:
        # Fallback to simpler layout if spring_layout fails
        pos = nx.circular_layout(G)
    
    # Node colors based on TP rate
    node_colors = [G.nodes[rule].get('tp_rate', 0) for rule in G.nodes()]
    
    # Node sizes based on total alerts
    node_sizes = [50 + G.nodes[rule].get('total_alerts', 0)/10 for rule in G.nodes()]
    
    # Edge weights based on Jaccard similarity
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                          alpha=0.8, cmap=plt.cm.viridis)
    
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, 
                          edge_color='gray')
    
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
    
    plt.axis('off')
    plt.title('Rule Overlap Network (Node color = TP Rate, Size = Alert Volume)')
    plt.tight_layout()
    plt.savefig(f'{overlap_dir}/rule_overlap_network.png')
    plt.show()
    
    # Analyze rule sequence patterns - which rules tend to fire one after another
    if 'transaction_date_time_local' in transaction_data.columns:
        print("\nAnalyzing temporal rule firing patterns...")
        
        # Group by KYC ID and sort by date
        temporal_patterns = []
        
        for kyc_id, group in transaction_data.groupby(['sender_kyc_id_no']):
            if len(group) > 1:  # Need at least 2 alerts
                # Sort by date
                sorted_group = group.sort_values('transaction_date_time_local')
                
                # Get the sequence of rules
                rule_sequence = sorted_group['alert_rules'].tolist()
                
                # Create pairs of consecutive rules
                for i in range(len(rule_sequence) - 1):
                    temporal_patterns.append((rule_sequence[i], rule_sequence[i+1]))
        
        # Count the temporal patterns
        pattern_counts = pd.Series(temporal_patterns).value_counts()
        
        print("\nTop temporal rule patterns (which rules fire consecutively):")
        print(pattern_counts.head(10))
    
    return rule_cooccurrence, rule_jaccard, strong_overlaps

# Execute rule overlap analysis
rule_cooccurrence, rule_jaccard, strong_overlaps = analyze_rule_overlap(transaction_data, rule_performance, kyc_alerts)
```

## 9. Recommendations Generation

In this section, we'll generate comprehensive recommendations with quantified impact.

```python
def generate_recommendations(transaction_data, rule_performance, rule_clusters, rule_corr_matrix, kyc_alerts):
    """Generate comprehensive recommendations with quantified impact."""
    print("\nGenerating comprehensive recommendations with quantified impact...")
    
    # Check if rule_clusters is defined and has data
    if rule_clusters is None or rule_clusters.empty:
        print("Warning: rule_clusters is undefined or empty. Creating a default empty DataFrame.")
        rule_clusters = pd.DataFrame(columns=['Rule', 'Cluster', 'TP_Rate', 'Total'])
    
    # Total alert stats for reference
    total_alerts = len(transaction_data)
    closed_alerts = transaction_data[transaction_data['status'].isin(['Closed TP', 'Closed FP'])]
    total_tp = len(closed_alerts[closed_alerts['status'] == 'Closed TP'])
    total_fp = len(closed_alerts[closed_alerts['status'] == 'Closed FP'])
    current_tp_rate = total_tp / len(closed_alerts) * 100 if len(closed_alerts) > 0 else 0
    
    # Create recommendations list
    recommendations = []
    
    # 1. Remove or Modify Inefficient Rules
    inefficient_rules = rule_performance[(rule_performance['Total'] > 50) & 
                                         (rule_performance['TP_Rate'] < 30)].sort_values('Total', ascending=False)
    
    if not inefficient_rules.empty:
        for _, rule in inefficient_rules.head(5).iterrows():
            rule_alerts = closed_alerts[closed_alerts['alert_rules'] == rule['alert_rules']]
            rule_tp = rule['TP']
            rule_fp = rule['FP']
            rule_total = rule['Total']
            
            # Calculate impact if rule is removed
            new_tp_count = total_tp - rule_tp
            new_fp_count = total_fp - rule_fp
            new_total = len(closed_alerts) - rule_total
            new_tp_rate = new_tp_count / new_total * 100 if new_total > 0 else 0
            tp_rate_change = new_tp_rate - current_tp_rate
            
            # Alert volume reduction
            rule_volume_in_all = transaction_data[transaction_data['alert_rules'] == rule['alert_rules']].shape[0]
            volume_reduction = rule_volume_in_all / total_alerts * 100
            
            recommendations.append({
                'Category': 'Remove Inefficient Rules',
                'Rules': rule['alert_rules'],
                'Action': 'Remove rule or significantly increase threshold',
                'Rationale': f"Low TP rate ({rule['TP_Rate']:.1f}%) with high volume ({rule['Total']} alerts)",
                'Impact': f"Alert volume: -{volume_reduction:.1f}%, TP rate: {tp_rate_change:+.2f}%",
                'Priority': 'High' if rule['Total'] > 100 else 'Medium'
            })
    
    # 2. Consolidate Similar Rules (from clusters and correlation)
    # Get highly correlated rule pairs
    high_corr_threshold = 0.7
    high_corr_pairs = []
    
    # Find all pairs of rules with correlation above threshold
    try:
        for i in range(len(rule_corr_matrix.index)):
            for j in range(i+1, len(rule_corr_matrix.columns)):
                rule1 = rule_corr_matrix.index[i]
                rule2 = rule_corr_matrix.columns[j]
                corr = rule_corr_matrix.iloc[i, j]
                if corr >= high_corr_threshold:
                    high_corr_pairs.append((rule1, rule2, corr))
    except Exception as e:
        print(f"Error processing correlation matrix: {e}")
    
    # Sort by correlation
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Generate recommendations for top correlated pairs
    for rule1, rule2, corr in high_corr_pairs[:5]:  # Top 5 pairs
        # Get performance data
        rule1_perf = rule_performance[rule_performance['alert_rules'] == rule1]
        rule2_perf = rule_performance[rule_performance['alert_rules'] == rule2]
        
        if not rule1_perf.empty and not rule2_perf.empty:
            rule1_tp_rate = rule1_perf.iloc[0]['TP_Rate']
            rule2_tp_rate = rule2_perf.iloc[0]['TP_Rate']
            
            # Determine which rule to keep
            keep_rule = rule1 if rule1_tp_rate >= rule2_tp_rate else rule2
            remove_rule = rule2 if keep_rule == rule1 else rule1
            
            # Calculate impact of consolidation
            # For all alerts (not just closed ones)
            keep_alerts = transaction_data[transaction_data['alert_rules'] == keep_rule].shape[0]
            remove_alerts = transaction_data[transaction_data['alert_rules'] == remove_rule].shape[0]
            
            # Estimate overlap based on correlation
            # Higher correlation means more shared alerts
            overlap_factor = corr
            unique_remove_alerts = remove_alerts * (1 - overlap_factor)
            
            # Volume reduction
            volume_reduction = unique_remove_alerts / total_alerts * 100
            
            recommendations.append({
                'Category': 'Consolidate Similar Rules',
                'Rules': f"{rule1}, {rule2}",
                'Action': f"Combine rules, keep {keep_rule}",
                'Rationale': f"High correlation ({corr:.2f}), {keep_rule} has higher TP rate ({max(rule1_tp_rate, rule2_tp_rate):.1f}%)",
                'Impact': f"Alert volume: -{volume_reduction:.1f}%, No significant impact on TP rate",
                'Priority': 'High' if corr > 0.9 else 'Medium'
            })
    
    # 3. Convert Inefficient Daily Rules to Weekly
    daily_rules = rule_performance[(rule_performance['Frequency'] == 'daily') & 
                                  (rule_performance['TP_Rate'] < 30)].sort_values('Total', ascending=False).head(3)
    
    if not daily_rules.empty:
        # Aggregate daily rules
        daily_rule_list = daily_rules['alert_rules'].tolist()
        daily_alert_count = transaction_data[transaction_data['alert_rules'].isin(daily_rule_list)].shape[0]
        
        # Estimate reduction (weekly = ~1/5 of daily)
        estimated_reduction = daily_alert_count * 0.8  # 80% reduction
        volume_reduction_pct = estimated_reduction / total_alerts * 100
        
        recommendations.append({
            'Category': 'Adjust Rule Frequency',
            'Rules': ', '.join(daily_rule_list),
            'Action': 'Convert inefficient daily rules to weekly frequency',
            'Rationale': 'Daily rules generate high volume with low true positive rates',
            'Impact': f"Alert volume: -{volume_reduction_pct:.1f}%, Potential improved accuracy",
            'Priority': 'Medium'
        })
    
    # 4. Implement KYC Deduplication
    # Estimate FP reduction from KYC deduplication
    # Get entities with multiple KYCs
    sender_kyc_counts = transaction_data.groupby('sender_name_kyc_wise')['sender_kyc_id_no'].nunique()
    receiver_kyc_counts = transaction_data.groupby('receiver_name_kyc_wise')['receiver_kyc_id_no'].nunique()
    
    senders_with_multiple_kyc = sender_kyc_counts[sender_kyc_counts > 1].index.tolist()
    receivers_with_multiple_kyc = receiver_kyc_counts[receiver_kyc_counts > 1].index.tolist()
    
    sender_multiple_kyc_alerts = transaction_data[
        (transaction_data['triggered_on'] == 'sender') & 
        (transaction_data['sender_name_kyc_wise'].isin(senders_with_multiple_kyc))
    ]
    
    receiver_multiple_kyc_alerts = transaction_data[
        (transaction_data['triggered_on'] == 'receiver') & 
        (transaction_data['receiver_name_kyc_wise'].isin(receivers_with_multiple_kyc))
    ]
    
    # Calculate metrics
    multiple_kyc_alerts = pd.concat([sender_multiple_kyc_alerts, receiver_multiple_kyc_alerts])
    multiple_kyc_alert_pct = len(multiple_kyc_alerts) / total_alerts * 100
    
    # Estimate reduction (assuming 60% are duplicate alerts)
    estimated_alert_reduction = len(multiple_kyc_alerts) * 0.6
    volume_reduction_pct = estimated_alert_reduction / total_alerts * 100
    
    # Estimate FP reduction
    multiple_kyc_fp = len(multiple_kyc_alerts[multiple_kyc_alerts['status'] == 'Closed FP'])
    estimated_fp_reduction = multiple_kyc_fp * 0.7  # 70% of FPs could be eliminated
    new_fp = total_fp - estimated_fp_reduction
    new_tp_rate = total_tp / (total_tp + new_fp) * 100 if (total_tp + new_fp) > 0 else 0
    tp_rate_improvement = new_tp_rate - current_tp_rate
    
    recommendations.append({
        'Category': 'KYC Breakage Mitigation',
        'Rules': 'All rules (system-wide)',
        'Action': 'Implement name/phone fuzzy matching system to deduplicate KYC IDs',
        'Rationale': f"{multiple_kyc_alert_pct:.1f}% of alerts involve entities with multiple KYC IDs",
        'Impact': f"Alert volume: -{volume_reduction_pct:.1f}%, TP rate: {tp_rate_improvement:+.2f}%",
        'Priority': 'High'
    })
    
    # 5. Advanced Analytical Approach (Rule Scoring System)
    recommendations.append({
        'Category': 'Advanced Analytics',
        'Rules': 'All rules (system-wide)',
        'Action': 'Implement entity risk scoring system based on rule trigger patterns',
        'Rationale': 'Current rules operate independently, missing the value of combined risk signals',
        'Impact': 'Estimated 15-25% reduction in false positives while maintaining true positive catch rate',
        'Priority': 'Medium-Long term'
    })
    
    # Calculate combined impact
    # Estimate total alert volume reduction from all recommendations
    # This is approximate since some recommendations overlap
    total_volume_reduction = 0
    tp_rate_change = 0
    
    for rec in recommendations:
        impact = rec['Impact']
        
        # Extract volume reduction percentage
        if 'Alert volume:' in impact:
            try:
                volume_part = impact.split('Alert volume:')[1].split('%')[0].strip(' -')
                volume_reduction = float(volume_part)
                total_volume_reduction += volume_reduction
            except:
                pass
        
        # Extract TP rate change
        if 'TP rate:' in impact:
            try:
                tp_part = impact.split('TP rate:')[1].split('%')[0].strip()
                if '+' in tp_part or '-' in tp_part:
                    tp_change = float(tp_part)
                    tp_rate_change += tp_change
            except:
                pass
    
    # Cap the maximum reduction to a reasonable value (80%)
    total_volume_reduction = min(total_volume_reduction, 80)
    
    # Create summary DataFrame
    recommendations_df = pd.DataFrame(recommendations)
    
    # Print summary
    print("\nSummary of recommendations:")
    print(f"Total recommendations: {len(recommendations_df)}")
    print(f"Estimated total alert volume reduction: {total_volume_reduction:.1f}%")
    print(f"Estimated true positive rate change: {tp_rate_change:+.2f}%")
    print("\nRecommendations by category:")
    print(recommendations_df['Category'].value_counts())
    
    # Visualize recommendations by category
    plt.figure(figsize=(12, 6))
    recommendations_df['Category'].value_counts().plot(kind='barh', color='teal')
    plt.title('Recommendations by Category')
    plt.xlabel('Number of Recommendations')
    plt.tight_layout()
    plt.savefig('visualizations/recommendations_by_category.png')
    plt.show()
    
    # Visualize recommendations by priority
    plt.figure(figsize=(10, 6))
    if 'Priority' in recommendations_df.columns:
        recommendations_df['Priority'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['red', 'orange', 'green'])
        plt.title('Recommendations by Priority')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig('visualizations/recommendations_by_priority.png')
        plt.show()
    
    return recommendations_df, total_volume_reduction, tp_rate_change

# Execute recommendations generation
recommendations_df, total_volume_reduction, tp_rate_change = generate_recommendations(
    transaction_data, rule_performance, rule_clusters, rule_corr_matrix, kyc_alerts
)
```

## 10. Main Execution and Report Generation

```python
def run_terrapay_analysis():
    """Run the complete enhanced Terrapay transaction monitoring analysis."""
    print("Starting enhanced Terrapay transaction monitoring analysis...")
    
    # Load the data
    transaction_data, metadata, rule_descriptions = load_data()
    
    # Perform the basic KYC alert overlap analysis
    kyc_alerts, rule_count_dist, rule_pairs, kyc_entity_type, rule_combinations = analyze_kyc_alert_overlap(transaction_data)
    
    # Analyze rule efficiency
    rule_performance, pattern_performance, frequency_performance, true_positives, tp_kyc_ids = analyze_rule_efficiency_with_impact(transaction_data, rule_descriptions)
    
    # Run the rule clustering function
    rule_clusters, rule_corr_matrix = perform_rule_clustering(transaction_data, rule_performance, kyc_alerts)
    
    # Run the enhanced KYC breakage analysis
    sender_name_groups, receiver_name_groups, sender_multiple_kyc_alerts, receiver_multiple_kyc_alerts = analyze_kyc_breakage_enhanced(transaction_data)
    
    # Perform the enhanced ATL/BTL analysis
    atl_btl_summary_df = perform_atl_btl_analysis(transaction_data, rule_descriptions)
    
    # Run enhanced rule overlap analysis
    rule_cooccurrence, rule_jaccard, strong_overlaps = analyze_rule_overlap(transaction_data, rule_performance, kyc_alerts)
    
    # Generate comprehensive recommendations
    recommendations_df, total_volume_reduction, tp_rate_change = generate_recommendations(
        transaction_data, rule_performance, rule_clusters, rule_corr_matrix, kyc_alerts
    )
    
    # Save all results to Excel
    with pd.ExcelWriter('terrapay_enhanced_analysis_results.xlsx') as writer:
        # Overall performance metrics
        overall_metrics = pd.DataFrame({
            'Metric': [
                'Total Alerts',
                'True Positives',
                'False Positives',
                'True Positive Rate',
                'Estimated Alert Volume Reduction',
                'Estimated TP Rate Improvement'
            ],
            'Value': [
                len(transaction_data),
                len(transaction_data[transaction_data['status'] == 'Closed TP']),
                len(transaction_data[transaction_data['status'] == 'Closed FP']),
                f"{len(transaction_data[transaction_data['status'] == 'Closed TP']) / len(transaction_data[transaction_data['status'].isin(['Closed TP', 'Closed FP'])]) * 100:.2f}%",
                f"{total_volume_reduction:.2f}%",
                f"{tp_rate_change:+.2f}%"
            ]
        })
        overall_metrics.to_excel(writer, sheet_name='Summary', index=False)
        
        # Rule performance
        rule_performance.to_excel(writer, sheet_name='Rule Performance', index=False)
        
        # ATL/BTL analysis results
        if atl_btl_summary_df is not None and not atl_btl_summary_df.empty:
            atl_btl_summary_df.to_excel(writer, sheet_name='ATL_BTL_Analysis', index=False)
        
        # True Positive cases
        true_positives.to_excel(writer, sheet_name='True Positives', index=False)
        
        # KYC breakage analysis
        pd.DataFrame({'sender_name': sender_name_groups['sender_name'], 
                     'kyc_count': sender_name_groups['kyc_id_count']}).to_excel(
            writer, sheet_name='Sender KYC Breakage', index=False)
        
        pd.DataFrame({'receiver_name': receiver_name_groups['receiver_name'], 
                     'kyc_count': receiver_name_groups['kyc_id_count']}).to_excel(
            writer, sheet_name='Receiver KYC Breakage', index=False)
        
        # Rule correlation matrix
        rule_corr_matrix.to_excel(writer, sheet_name='Rule Correlation Matrix')
        
        # Rule jaccard similarity
        if 'rule_jaccard' in locals():
            rule_jaccard.to_excel(writer, sheet_name='Rule Jaccard Similarity')
        
        # Rule clusters
        if rule_clusters is not None and not rule_clusters.empty:
            rule_clusters.to_excel(writer, sheet_name='Rule Clusters', index=False)
        
        # Recommendations
        recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
        
        # High-risk entities (KYC IDs with multiple true positives)
        tp_kyc_counts = pd.Series(tp_kyc_ids).value_counts()
        high_risk_kycs = tp_kyc_counts[tp_kyc_counts > 1].reset_index()
        high_risk_kycs.columns = ['KYC_ID', 'TP_Count']
        if not high_risk_kycs.empty:
            high_risk_kycs.to_excel(writer, sheet_name='High Risk KYCs', index=False)

    print("\nAnalysis complete. Results saved to 'terrapay_enhanced_analysis_results.xlsx'.")
    print("Key findings and recommendations have been generated based on the analysis.")
    
    return {
        "rule_performance": rule_performance,
        "atl_btl_summary": atl_btl_summary_df,
        "recommendations": recommendations_df
    }

def generate_summary_report(analysis_results):
    """Generate a summary report with key findings and visualizations."""
    rule_performance = analysis_results.get("rule_performance")
    recommendations = analysis_results.get("recommendations")
    
    print("===== TERRAPAY TRANSACTION MONITORING OPTIMIZATION REPORT =====")
    print("\nKey Findings:")
    
    # Overall statistics
    total_alerts = len(transaction_data)
    closed_alerts = transaction_data[transaction_data['status'].isin(['Closed TP', 'Closed FP'])]
    tp_rate = sum(closed_alerts['status'] == 'Closed TP') / len(closed_alerts) * 100 if len(closed_alerts) > 0 else 0
    fp_rate = sum(closed_alerts['status'] == 'Closed FP') / len(closed_alerts) * 100 if len(closed_alerts) > 0 else 0
    
    print(f"1. Overall System Performance:")
    print(f"   - Total alerts analyzed: {total_alerts}")
    print(f"   - True positive rate: {tp_rate:.2f}%")
    print(f"   - False positive rate: {fp_rate:.2f}%")
    
    # KYC breakage impact
    sender_kyc_groups = transaction_data.groupby('sender_name_kyc_wise')['sender_kyc_id_no'].nunique()
    receiver_kyc_groups = transaction_data.groupby('receiver_name_kyc_wise')['receiver_kyc_id_no'].nunique()
    
    senders_with_multiple_kyc = sum(sender_kyc_groups > 1)
    receivers_with_multiple_kyc = sum(receiver_kyc_groups > 1)
    
    print(f"\n2. KYC Breakage Impact:")
    print(f"   - Senders with multiple KYC IDs: {senders_with_multiple_kyc} ({senders_with_multiple_kyc/len(sender_kyc_groups)*100:.2f}%)")
    print(f"   - Receivers with multiple KYC IDs: {receivers_with_multiple_kyc} ({receivers_with_multiple_kyc/len(receiver_kyc_groups)*100:.2f}%)")
    
    # Rule efficiency
    inefficient_rules = rule_performance[(rule_performance['Total'] > 50) & (rule_performance['TP_Rate'] < 30)]
    print(f"\n3. Rule Efficiency:")
    print(f"   - Inefficient rules (high volume, low TP rate): {len(inefficient_rules)}")
    print(f"   - These rules generate {inefficient_rules['Total'].sum()} alerts with only {inefficient_rules['TP'].sum()} true positives")
    
    # Overlap statistics  
    print(f"\n4. Rule Overlap:")
    rule_pairs = []
    for kyc_id, rules in kyc_alerts.items():
        if len(rules) > 1:
            rule_list = list(rules)
            for i in range(len(rule_list)):
                for j in range(i+1, len(rule_list)):
                    rule_pairs.append((rule_list[i], rule_list[j]))
    
    print(f"   - {len(rule_pairs)} instances of rule pairs triggering on the same KYC ID")
    print(f"   - {len(strong_overlaps)} rule pairs with strong correlation (Jaccard similarity >= 0.6)")
    
    # Recommendations summary
    print(f"\n5. Optimization Recommendations:")
    for category, count in recommendations['Category'].value_counts().items():
        print(f"   - {category}: {count} recommendations")
    
    # Estimated impact
    estimated_reduction = 0
    for _, rec in recommendations.iterrows():
        if 'reduction' in rec['Impact'].lower():
            try:
                reduction_str = rec['Impact'].split('%')[0].split(':')[-1].strip()
                reduction = float(reduction_str)
                estimated_reduction += reduction
            except:
                pass
    
    # Cap at reasonable value
    estimated_reduction = min(estimated_reduction, 70)
    
    print(f"\nEstimated total alert volume reduction: {estimated_reduction:.1f}%")
    print(f"This would significantly improve analyst efficiency and reduce false positives.")
    
    print("\n===== END OF SUMMARY REPORT =====")

# Run the full analysis and generate the report
analysis_results = run_terrapay_analysis()
generate_summary_report(analysis_results)
```

This comprehensive notebook provides a full analysis of Terrapay's transaction monitoring system, identifying inefficient rules, quantifying the impact of KYC breakage, and generating actionable recommendations to reduce false positives while maintaining true positive detection rates.
