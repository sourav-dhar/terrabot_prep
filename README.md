def analyze_rule_overlap_matrix(transaction_data):
    """
    Create a detailed rule overlap matrix and visualization showing 
    how KYC IDs overlap between different rules.
    """
    print("\nAnalyzing detailed rule overlap matrix...")
    
    # Create a dictionary mapping KYC IDs to the rules they triggered
    kyc_alerts = defaultdict(set)
    
    for idx, row in transaction_data.iterrows():
        if row['triggered_on'] == 'sender':
            kyc_id = row['sender_kyc_id_no']
        else:  # receiver
            kyc_id = row['receiver_kyc_id_no']
            
        kyc_alerts[kyc_id].add(row['alert_rules'])
    
    # Get unique rules
    all_rules = sorted(transaction_data['alert_rules'].unique())
    
    # Create a matrix to store overlap counts
    overlap_matrix = pd.DataFrame(0, index=all_rules, columns=all_rules)
    
    # Calculate the number of KYC IDs that triggered each pair of rules
    for kyc_id, rules in kyc_alerts.items():
        if len(rules) > 1:
            # For each pair of rules, increment the overlap count
            rule_list = list(rules)
            for i in range(len(rule_list)):
                for j in range(len(rule_list)):
                    rule_i, rule_j = rule_list[i], rule_list[j]
                    overlap_matrix.loc[rule_i, rule_j] += 1
    
    # For the diagonal (self-overlap), count the total number of KYC IDs for each rule
    for rule in all_rules:
        rule_kyc_ids = sum(1 for kyc_id, rules in kyc_alerts.items() if rule in rules)
        overlap_matrix.loc[rule, rule] = rule_kyc_ids
    
    # Calculate the overlap percentages (normalize by the diagonal)
    overlap_pct = pd.DataFrame(0.0, index=all_rules, columns=all_rules)
    for i in all_rules:
        total_i = overlap_matrix.loc[i, i]
        if total_i > 0:
            for j in all_rules:
                overlap_pct.loc[i, j] = (overlap_matrix.loc[i, j] / total_i) * 100
    
    # Create visualizations
    # 1. Heatmap of absolute overlap counts
    plt.figure(figsize=(16, 14))
    sns.heatmap(overlap_matrix, cmap="YlGnBu", annot=False)
    plt.title('Rule Overlap Matrix (Absolute Counts)')
    plt.tight_layout()
    plt.savefig('visualizations/rule_overlap_matrix_absolute.png')
    plt.close()
    
    # 2. Heatmap of overlap percentages
    plt.figure(figsize=(16, 14))
    sns.heatmap(overlap_pct, cmap="YlGnBu", annot=False, vmin=0, vmax=100)
    plt.title('Rule Overlap Matrix (% of Row Rule)')
    plt.tight_layout()
    plt.savefig('visualizations/rule_overlap_matrix_percentage.png')
    
    # 3. Create a subset heatmap with only the most frequent rules for better readability
    # Get the top 20 rules by frequency
    rule_counts = [overlap_matrix.loc[rule, rule] for rule in all_rules]
    top_rules_idx = np.argsort(rule_counts)[-20:]
    top_rules = [all_rules[i] for i in top_rules_idx]
    
    # Create subset matrices
    top_matrix = overlap_matrix.loc[top_rules, top_rules]
    top_pct = overlap_pct.loc[top_rules, top_rules]
    
    # Plot the subset heatmap for percentages
    plt.figure(figsize=(14, 12))
    sns.heatmap(top_pct, cmap="YlGnBu", annot=True, fmt='.1f', vmin=0, vmax=100)
    plt.title('Rule Overlap Matrix - Top 20 Rules (% of Row Rule)')
    plt.tight_layout()
    plt.savefig('visualizations/rule_overlap_matrix_top20_percentage.png')
    plt.show()
    
    # Extract the most significant overlaps (high percentage)
    high_overlap_threshold = 80  # 80% overlap
    high_overlaps = []
    
    for i in all_rules:
        for j in all_rules:
            if i != j and overlap_pct.loc[i, j] >= high_overlap_threshold:
                high_overlaps.append((i, j, overlap_pct.loc[i, j]))
    
    # Sort by overlap percentage
    high_overlaps.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nFound {len(high_overlaps)} significant rule overlaps (>= {high_overlap_threshold}% overlap):")
    for rule1, rule2, pct in high_overlaps[:10]:  # Show top 10
        count = overlap_matrix.loc[rule1, rule2]
        total = overlap_matrix.loc[rule1, rule1]
        print(f"{rule1} → {rule2}: {count} KYCs ({pct:.1f}% of {rule1}'s {total} KYCs)")
    
    # Create a directed graph visualization of significant overlaps
    if high_overlaps:
        G = nx.DiGraph()
        
        # Add nodes
        for rule in all_rules:
            count = overlap_matrix.loc[rule, rule]
            G.add_node(rule, count=count)
        
        # Add edges for significant overlaps
        for rule1, rule2, pct in high_overlaps:
            G.add_edge(rule1, rule2, weight=pct)
        
        # Create visualization if not too large
        if len(G.nodes) <= 50:  # Limit for readability
            plt.figure(figsize=(16, 14))
            
            # Position nodes using spring layout
            pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
            
            # Get node sizes based on count (scaled)
            counts = [G.nodes[rule]['count'] for rule in G.nodes()]
            max_count = max(counts) if counts else 1
            node_sizes = [50 + (count / max_count) * 500 for count in counts]
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                  node_color='skyblue', alpha=0.8)
            
            # Draw edges with varying width and color based on overlap percentage
            edges = G.edges(data=True)
            edge_colors = [e[2]['weight'] / 100 for e in edges]
            edge_widths = [e[2]['weight'] / 20 for e in edges]
            
            nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                                  edge_cmap=plt.cm.Reds, alpha=0.7, 
                                  arrowsize=20, arrowstyle='->')
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10)
            
            plt.title('Rule Overlap Network (Edge color intensity = overlap %)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('visualizations/rule_overlap_network_directed.png')
            plt.show()
    
    return overlap_matrix, overlap_pct, high_overlaps


def analyze_daily_weekly_monthly_rule_coverage(transaction_data, rule_descriptions):
    """
    Analyze daily rules to see if they're covered by weekly or monthly versions.
    """
    print("\nAnalyzing daily, weekly, and monthly rule coverage...")
    
    # Extract the frequency information for each rule
    rule_freq_mapping = {}
    rule_desc_mapping = {}
    
    # Create mapping of rule to frequency and description
    for _, row in rule_descriptions.iterrows():
        if 'Rule no.' in row and 'Frequency' in row and 'Rule description' in row:
            rule = row['Rule no.']
            rule_freq_mapping[rule] = row['Frequency']
            rule_desc_mapping[rule] = row['Rule description']
    
    # Extract rule patterns (TRP_XXXX where XXXX is the base rule number)
    rule_patterns = {}
    for rule in rule_freq_mapping.keys():
        # Extract the basic rule pattern (removes frequency indicators)
        if '_' in rule:
            parts = rule.split('_')
            if len(parts) >= 2:
                try:
                    pattern = int(parts[1]) if parts[1].isdigit() else parts[1]
                    rule_patterns[rule] = pattern
                except:
                    rule_patterns[rule] = rule
        else:
            rule_patterns[rule] = rule
    
    # Group rules by pattern and frequency
    pattern_freq_rules = defaultdict(lambda: defaultdict(list))
    
    for rule, pattern in rule_patterns.items():
        freq = rule_freq_mapping.get(rule, 'Unknown')
        pattern_freq_rules[pattern][freq].append(rule)
    
    # Identify rule families with daily + weekly/monthly versions
    rule_families = []
    
    for pattern, freq_dict in pattern_freq_rules.items():
        # If the pattern has both daily and weekly/monthly versions
        if 'daily' in freq_dict and ('Weekly' in freq_dict or 'Monthly' in freq_dict):
            daily_rules = freq_dict['daily']
            weekly_rules = freq_dict.get('Weekly', [])
            monthly_rules = freq_dict.get('Monthly', [])
            
            rule_families.append({
                'Pattern': pattern,
                'Daily Rules': daily_rules,
                'Weekly Rules': weekly_rules,
                'Monthly Rules': monthly_rules
            })
    
    print(f"\nFound {len(rule_families)} rule families with daily + weekly/monthly versions:")
    
    # For each rule family, analyze KYC overlap
    family_analyses = []
    
    for family in rule_families:
        print(f"\nAnalyzing rule family with pattern {family['Pattern']}:")
        print(f"  Daily rules: {', '.join(family['Daily Rules'])}")
        print(f"  Weekly rules: {', '.join(family['Weekly Rules'])}")
        print(f"  Monthly rules: {', '.join(family['Monthly Rules'])}")
        
        # Get the KYC IDs for each rule
        daily_kycs = set()
        weekly_kycs = set()
        monthly_kycs = set()
        
        for idx, row in transaction_data.iterrows():
            rule = row['alert_rules']
            if row['triggered_on'] == 'sender':
                kyc_id = row['sender_kyc_id_no']
            else:  # receiver
                kyc_id = row['receiver_kyc_id_no']
            
            if rule in family['Daily Rules']:
                daily_kycs.add(kyc_id)
            elif rule in family['Weekly Rules']:
                weekly_kycs.add(kyc_id)
            elif rule in family['Monthly Rules']:
                monthly_kycs.add(kyc_id)
        
        # Calculate overlap statistics
        daily_total = len(daily_kycs)
        weekly_total = len(weekly_kycs)
        monthly_total = len(monthly_kycs)
        
        # Daily in weekly overlap
        daily_in_weekly = len(daily_kycs.intersection(weekly_kycs))
        daily_in_weekly_pct = (daily_in_weekly / daily_total * 100) if daily_total > 0 else 0
        
        # Daily in monthly overlap
        daily_in_monthly = len(daily_kycs.intersection(monthly_kycs))
        daily_in_monthly_pct = (daily_in_monthly / daily_total * 100) if daily_total > 0 else 0
        
        # Daily in weekly OR monthly (total coverage)
        daily_in_any = len(daily_kycs.intersection(weekly_kycs.union(monthly_kycs)))
        daily_in_any_pct = (daily_in_any / daily_total * 100) if daily_total > 0 else 0
        
        # Daily unique (not in weekly or monthly)
        daily_unique = len(daily_kycs - weekly_kycs.union(monthly_kycs))
        daily_unique_pct = (daily_unique / daily_total * 100) if daily_total > 0 else 0
        
        print(f"  Daily KYCs: {daily_total}, Weekly KYCs: {weekly_total}, Monthly KYCs: {monthly_total}")
        print(f"  Daily KYCs also in weekly: {daily_in_weekly} ({daily_in_weekly_pct:.1f}%)")
        print(f"  Daily KYCs also in monthly: {daily_in_monthly} ({daily_in_monthly_pct:.1f}%)")
        print(f"  Daily KYCs in either weekly or monthly: {daily_in_any} ({daily_in_any_pct:.1f}%)")
        print(f"  Daily KYCs unique (not in weekly/monthly): {daily_unique} ({daily_unique_pct:.1f}%)")
        
        # Add to our analysis collection
        family_analyses.append({
            'Pattern': family['Pattern'],
            'Daily Rules': family['Daily Rules'],
            'Weekly Rules': family['Weekly Rules'],
            'Monthly Rules': family['Monthly Rules'],
            'Daily Total': daily_total,
            'Weekly Total': weekly_total,
            'Monthly Total': monthly_total,
            'Daily in Weekly': daily_in_weekly,
            'Daily in Weekly %': daily_in_weekly_pct,
            'Daily in Monthly': daily_in_monthly,
            'Daily in Monthly %': daily_in_monthly_pct,
            'Daily in Any': daily_in_any,
            'Daily in Any %': daily_in_any_pct,
            'Daily Unique': daily_unique,
            'Daily Unique %': daily_unique_pct
        })
    
    # Create a summary DataFrame
    family_df = pd.DataFrame(family_analyses)
    
    # Plot the coverage analysis
    if not family_df.empty:
        # Sort by daily unique percentage (ascending)
        family_df_sorted = family_df.sort_values('Daily Unique %')
        
        plt.figure(figsize=(14, 8))
        
        # Create a stacked bar chart
        bars = plt.barh(family_df_sorted['Pattern'].astype(str), family_df_sorted['Daily Unique %'], 
                     label='Daily Unique %', color='red', alpha=0.7)
        
        plt.barh(family_df_sorted['Pattern'].astype(str), family_df_sorted['Daily in Any %'], 
               label='Daily in Weekly/Monthly %', color='green', alpha=0.7, left=0)
        
        # Add value labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                   f"{family_df_sorted['Daily Unique %'].iloc[i]:.1f}%", 
                   va='center')
        
        plt.xlabel('Percentage of Daily Rule KYCs')
        plt.ylabel('Rule Pattern')
        plt.title('Coverage of Daily Rules by Weekly/Monthly Versions')
        plt.legend(loc='upper right')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/daily_rules_coverage.png')
        plt.show()
        
        # Generate recommendations
        recommendations = []
        
        for _, row in family_df.iterrows():
            pattern = row['Pattern']
            daily_rules = row['Daily Rules']
            
            if row['Daily Unique %'] < 20:  # Less than 20% unique coverage
                recommendation = {
                    'Pattern': pattern,
                    'Daily Rules': daily_rules,
                    'Action': 'Consider removing',
                    'Rationale': f"Only {row['Daily Unique %']:.1f}% of daily alerts aren't caught by weekly/monthly rules",
                    'Impact': f"Reduction of {row['Daily Total']} alerts with minimal loss of coverage"
                }
                recommendations.append(recommendation)
            elif row['Daily Unique %'] < 50:  # 20-50% unique coverage
                recommendation = {
                    'Pattern': pattern,
                    'Daily Rules': daily_rules,
                    'Action': 'Consider adjusting thresholds',
                    'Rationale': f"{row['Daily Unique %']:.1f}% of daily alerts provide unique coverage",
                    'Impact': "Potential reduction in alert volume while maintaining critical coverage"
                }
                recommendations.append(recommendation)
            else:  # >50% unique coverage
                recommendation = {
                    'Pattern': pattern,
                    'Daily Rules': daily_rules,
                    'Action': 'Retain but monitor',
                    'Rationale': f"High unique coverage ({row['Daily Unique %']:.1f}% not caught by weekly/monthly)",
                    'Impact': "Important for timely detection"
                }
                recommendations.append(recommendation)
        
        # Create recommendations DataFrame
        recommendations_df = pd.DataFrame(recommendations)
        
        if not recommendations_df.empty:
            print("\nRecommendations for daily rules:")
            print(recommendations_df[['Pattern', 'Daily Rules', 'Action', 'Rationale']])
    
    return family_df, recommendations_df if 'recommendations_df' in locals() else None

# Execute the rule overlap matrix analysis
overlap_matrix, overlap_pct, high_overlaps = analyze_rule_overlap_matrix(transaction_data)

# Execute the daily/weekly/monthly rule coverage analysis
family_df, daily_rule_recommendations = analyze_daily_weekly_monthly_rule_coverage(transaction_data, rule_descriptions)

# Let's also identify the most redundant rules based on the overlap matrix
def identify_most_redundant_rules(overlap_pct, min_overlap=80, min_count=5):
    """Identify the most redundant rules based on the overlap percentage matrix."""
    rule_redundancy = {}
    
    # For each rule, count how many other rules it significantly overlaps with
    for rule in overlap_pct.index:
        # Count rules that have high overlap with this rule
        overlapping_rules = sum(1 for other_rule in overlap_pct.columns 
                              if rule != other_rule and overlap_pct.loc[rule, other_rule] >= min_overlap)
        
        # Also get the total KYC count for this rule
        rule_count = overlap_pct.loc[rule, rule]
        
        if overlapping_rules > 0 and rule_count >= min_count:
            rule_redundancy[rule] = {
                'Overlapping_Rules': overlapping_rules,
                'KYC_Count': rule_count
            }
    
    # Sort by number of overlapping rules (descending)
    sorted_redundancy = sorted(rule_redundancy.items(), 
                              key=lambda x: x[1]['Overlapping_Rules'], 
                              reverse=True)
    
    print("\nTop 10 most redundant rules (highest overlap with other rules):")
    for rule, stats in sorted_redundancy[:10]:
        print(f"{rule}: Overlaps with {stats['Overlapping_Rules']} other rules, has {stats['KYC_Count']} KYCs")
    
    return sorted_redundancy

# Execute the redundant rule analysis
redundant_rules = identify_most_redundant_rules(overlap_pct)
==========================================================================
def perform_enhanced_rule_clustering(transaction_data, rule_performance, kyc_alerts):
    """
    Perform comprehensive rule clustering analysis with improved visualizations
    and business-oriented explanations.
    """
    print("\nPerforming enhanced rule clustering analysis...")
    
    # Create directories for cluster visualizations
    cluster_dir = 'visualizations/clustering'
    os.makedirs(cluster_dir, exist_ok=True)
    
    # Extract rule co-occurrence patterns
    all_rules = sorted(transaction_data['alert_rules'].unique())
    rule_matrix = pd.DataFrame(0, index=kyc_alerts.keys(), columns=all_rules)
    
    for kyc_id, rules in kyc_alerts.items():
        for rule in rules:
            if rule in all_rules:  # Ensure rule is in the columns
                rule_matrix.loc[kyc_id, rule] = 1
    
    # Calculate correlation matrix between rules
    rule_corr_matrix = rule_matrix.corr()
    rule_corr_matrix.fillna(0, inplace=True)  # Handle NaN values
    
    # Create a distance matrix (1 - correlation)
    distance_matrix = 1 - rule_corr_matrix.abs()
    
    # Extract rule characteristics for profiling
    rule_profiles = pd.DataFrame(index=all_rules)
    
    # Add performance metrics to profiles
    for rule in all_rules:
        rule_data = rule_performance[rule_performance['alert_rules'] == rule]
        if not rule_data.empty:
            rule_profiles.loc[rule, 'TP_Rate'] = rule_data['TP_Rate'].iloc[0]
            rule_profiles.loc[rule, 'FP_Rate'] = rule_data['FP_Rate'].iloc[0] if 'FP_Rate' in rule_data.columns else None
            rule_profiles.loc[rule, 'Total_Alerts'] = rule_data['Total'].iloc[0]
            rule_profiles.loc[rule, 'Pattern'] = rule_data['Pattern'].iloc[0]
            rule_profiles.loc[rule, 'Frequency'] = rule_data['Frequency'].iloc[0]
        else:
            # Default values if rule not found
            rule_profiles.loc[rule, 'TP_Rate'] = 0
            rule_profiles.loc[rule, 'FP_Rate'] = 0
            rule_profiles.loc[rule, 'Total_Alerts'] = 0
            rule_profiles.loc[rule, 'Pattern'] = 'Unknown'
            rule_profiles.loc[rule, 'Frequency'] = 'Unknown'
    
    # Convert to numeric where appropriate
    rule_profiles['TP_Rate'] = pd.to_numeric(rule_profiles['TP_Rate'], errors='coerce')
    rule_profiles['FP_Rate'] = pd.to_numeric(rule_profiles['FP_Rate'], errors='coerce')
    rule_profiles['Total_Alerts'] = pd.to_numeric(rule_profiles['Total_Alerts'], errors='coerce')
    
    # Fill missing values
    rule_profiles.fillna({'TP_Rate': 0, 'FP_Rate': 0, 'Total_Alerts': 0, 
                         'Pattern': 'Unknown', 'Frequency': 'Unknown'}, inplace=True)
    
    try:
        # 1. Determine optimal number of clusters using silhouette analysis
        print("\nDetermining optimal number of clusters...")
        range_n_clusters = list(range(2, min(11, len(all_rules))))  # 2 to 10 clusters
        silhouette_scores = []
        
        for n_clusters in range_n_clusters:
            # Initialize the clusterer
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            cluster_labels = clusterer.fit_predict(distance_matrix)
            
            # Calculate silhouette score
            if len(set(cluster_labels)) > 1:  # Check that we have more than one cluster
                silhouette_avg = silhouette_score(distance_matrix, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                print(f"  For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg:.3f}")
            else:
                silhouette_scores.append(0)
        
        # Determine optimal number of clusters (highest silhouette score)
        if silhouette_scores:
            optimal_clusters = range_n_clusters[np.argmax(silhouette_scores)]
            print(f"\nOptimal number of clusters: {optimal_clusters}")
        else:
            optimal_clusters = min(8, len(all_rules))  # Default if silhouette analysis fails
            print(f"\nUsing default number of clusters: {optimal_clusters}")
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(range_n_clusters, silhouette_scores, 'o-', color='blue')
        plt.axvline(x=optimal_clusters, color='red', linestyle='--')
        plt.title('Silhouette Score Method for Optimal Cluster Count')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{cluster_dir}/silhouette_analysis.png')
        plt.close()
        
        # 2. Perform hierarchical clustering with optimal number of clusters
        print(f"\nPerforming hierarchical clustering with {optimal_clusters} clusters...")
        linkage_matrix = linkage(distance_matrix.values, method='ward')
        
        # Plot dendrogram with cleaner design
        plt.figure(figsize=(14, 8))
        plt.title(f"Hierarchical Clustering of Transaction Monitoring Rules (Optimal: {optimal_clusters} clusters)")
        
        # Draw rectangle around clusters
        dendrogram(
            linkage_matrix,
            labels=all_rules,
            leaf_rotation=90,
            leaf_font_size=8,
            color_threshold=linkage_matrix[-optimal_clusters+1, 2],  # Color threshold to get optimal_clusters
            above_threshold_color='grey'
        )
        
        plt.tight_layout()
        plt.savefig(f'{cluster_dir}/rule_hierarchical_clustering.png')
        plt.close()
        
        # Perform the clustering with optimal number
        cluster_model = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
        cluster_labels = cluster_model.fit_predict(distance_matrix)
        
        # Create a DataFrame with rule clusters
        rule_clusters = pd.DataFrame({
            'Rule': all_rules,
            'Cluster': cluster_labels
        })
        
        # Add rule profiles to the clusters
        rule_clusters = rule_clusters.merge(rule_profiles, left_on='Rule', right_index=True)
        
        # 3. Generate 2D visualization using PCA for dimensionality reduction
        from sklearn.decomposition import PCA
        
        # Create a PCA model to reduce to 2 dimensions
        pca = PCA(n_components=2)
        rule_pca = pca.fit_transform(distance_matrix)
        
        # Create a clean scatter plot of rules in 2D space
        plt.figure(figsize=(12, 10))
        
        # Define a color palette
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, optimal_clusters))
        
        # Plot each cluster with a different color
        for cluster_id in range(optimal_clusters):
            cluster_points = rule_pca[cluster_labels == cluster_id]
            cluster_rules = rule_clusters[rule_clusters['Cluster'] == cluster_id]['Rule'].values
            
            # Plot points
            plt.scatter(
                cluster_points[:, 0], 
                cluster_points[:, 1],
                s=80, 
                color=cluster_colors[cluster_id],
                alpha=0.7,
                label=f'Cluster {cluster_id+1}'
            )
            
            # Add rule labels with smaller font and slight offset
            for i, rule in enumerate(cluster_rules):
                plt.annotate(
                    rule,
                    (rule_pca[all_rules.index(rule), 0], rule_pca[all_rules.index(rule), 1]),
                    fontsize=8,
                    alpha=0.8,
                    xytext=(5, 5),
                    textcoords='offset points'
                )
        
        plt.title('Rule Clusters Visualization (PCA Projection)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend(loc='upper right', title='Clusters')
        plt.grid(True, alpha=0.2)
        plt.savefig(f'{cluster_dir}/rule_clusters_2d.png')
        plt.close()
        
        # 4. Analyze and profile each cluster
        print("\nAnalyzing clusters...")
        cluster_profiles = []
        
        for cluster_id in range(optimal_clusters):
            # Get rules in this cluster
            cluster_rules = rule_clusters[rule_clusters['Cluster'] == cluster_id]
            
            # Calculate metrics for this cluster
            avg_tp_rate = cluster_rules['TP_Rate'].mean()
            avg_fp_rate = cluster_rules['FP_Rate'].mean() if 'FP_Rate' in cluster_rules.columns else None
            total_alerts = cluster_rules['Total_Alerts'].sum()
            
            # Get most common patterns and frequencies
            pattern_counts = cluster_rules['Pattern'].value_counts()
            dominant_pattern = pattern_counts.index[0] if not pattern_counts.empty else 'Unknown'
            pattern_diversity = len(pattern_counts)
            
            freq_counts = cluster_rules['Frequency'].value_counts()
            dominant_freq = freq_counts.index[0] if not freq_counts.empty else 'Unknown'
            freq_diversity = len(freq_counts)
            
            # Calculate alert volume proportion
            volume_proportion = total_alerts / rule_clusters['Total_Alerts'].sum() * 100
            
            # Define cluster type based on characteristics
            if avg_tp_rate >= 70:
                cluster_type = "High-Performance"
                recommendation = "Maintain these high-performing rules"
            elif avg_tp_rate >= 40:
                cluster_type = "Medium-Performance"
                recommendation = "Monitor and tune thresholds periodically"
            elif total_alerts > rule_clusters['Total_Alerts'].mean() * 2:
                cluster_type = "High-Volume, Low-Performance"
                recommendation = "Consider removing or significantly adjusting thresholds"
            else:
                cluster_type = "Low-Performance"
                recommendation = "Evaluate value and consider consolidation"
            
            # Create profile
            profile = {
                'Cluster_ID': cluster_id + 1,  # 1-indexed for business users
                'Rule_Count': len(cluster_rules),
                'Total_Alerts': total_alerts,
                'Volume_Proportion': volume_proportion,
                'Avg_TP_Rate': avg_tp_rate,
                'Avg_FP_Rate': avg_fp_rate,
                'Dominant_Pattern': dominant_pattern,
                'Pattern_Diversity': pattern_diversity,
                'Dominant_Frequency': dominant_freq,
                'Frequency_Diversity': freq_diversity,
                'Cluster_Type': cluster_type,
                'Recommendation': recommendation,
                'Rules': cluster_rules['Rule'].tolist()
            }
            
            cluster_profiles.append(profile)
        
        # Create cluster profiles DataFrame
        cluster_profiles_df = pd.DataFrame(cluster_profiles)
        
        # 5. Create visual summary of clusters
        # Bar chart comparing clusters by TP rate and volume
        plt.figure(figsize=(12, 6))
        
        # Sort by TP rate
        sorted_profiles = cluster_profiles_df.sort_values('Avg_TP_Rate', ascending=False)
        
        # Create bar chart
        ax = plt.subplot(111)
        bars = ax.bar(
            [f"Cluster {c}" for c in sorted_profiles['Cluster_ID']],
            sorted_profiles['Avg_TP_Rate'],
            color=[plt.cm.viridis(x/100) for x in sorted_profiles['Avg_TP_Rate']],
            alpha=0.7
        )
        
        # Add a second y-axis for volume proportion
        ax2 = ax.twinx()
        ax2.plot(
            [f"Cluster {c}" for c in sorted_profiles['Cluster_ID']],
            sorted_profiles['Volume_Proportion'],
            'o-',
            color='red',
            alpha=0.7,
            linewidth=2,
            markersize=8
        )
        
        # Add value labels
        for i, bar in enumerate(bars):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{sorted_profiles['Avg_TP_Rate'].iloc[i]:.1f}%",
                ha='center',
                va='bottom',
                fontsize=9
            )
            
            # Add volume label
            ax2.text(
                i,
                sorted_profiles['Volume_Proportion'].iloc[i] + 1,
                f"{sorted_profiles['Volume_Proportion'].iloc[i]:.1f}%",
                ha='center',
                va='bottom',
                color='red',
                fontsize=9
            )
        
        # Add cluster size as text below x-axis
        for i, cluster_id in enumerate(sorted_profiles['Cluster_ID']):
            rule_count = sorted_profiles[sorted_profiles['Cluster_ID'] == cluster_id]['Rule_Count'].iloc[0]
            plt.text(
                i, 
                -5, 
                f"{rule_count} rules",
                ha='center',
                fontsize=8
            )
        
        ax.set_title('Cluster Performance Comparison')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Average True Positive Rate (%)')
        ax2.set_ylabel('Alert Volume Proportion (%)', color='red')
        ax2.tick_params(axis='y', colors='red')
        
        # Adjust y-axis to start from 0
        ax.set_ylim(0, max(sorted_profiles['Avg_TP_Rate']) * 1.1)
        ax2.set_ylim(0, max(sorted_profiles['Volume_Proportion']) * 1.1)
        
        plt.tight_layout()
        plt.savefig(f'{cluster_dir}/cluster_performance_comparison.png')
        plt.close()
        
        # 6. Create multi-dimensional radar charts for each cluster
        # First, prepare data for radar charts
        categories = ['TP Rate', 'Alert Volume', 'Pattern Diversity', 'Frequency Diversity']
        N = len(categories)
        
        # Create a figure with subplots for radar charts
        fig = plt.figure(figsize=(15, 10))
        
        # Calculate maximum values for normalization
        max_tp = max(cluster_profiles_df['Avg_TP_Rate'])
        max_volume = max(cluster_profiles_df['Volume_Proportion'])
        max_pattern = max(cluster_profiles_df['Pattern_Diversity'])
        max_freq = max(cluster_profiles_df['Frequency_Diversity'])
        
        # Create subplot grid
        rows = (optimal_clusters + 2) // 3  # Calculate rows needed (3 per row)
        cols = min(3, optimal_clusters)     # Maximum 3 columns
        
        for i, profile in enumerate(cluster_profiles):
            cluster_id = profile['Cluster_ID']
            
            # Normalize values between 0 and 1
            values = [
                profile['Avg_TP_Rate'] / max_tp if max_tp > 0 else 0,
                profile['Volume_Proportion'] / max_volume if max_volume > 0 else 0,
                profile['Pattern_Diversity'] / max_pattern if max_pattern > 0 else 0,
                profile['Frequency_Diversity'] / max_freq if max_freq > 0 else 0
            ]
            
            # Close the loop
            values += values[:1]
            
            # Calculate angle for each category
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Create subplot
            ax = fig.add_subplot(rows, cols, i+1, polar=True)
            
            # Plot data
            ax.plot(angles, values, linewidth=2, linestyle='solid', color=cluster_colors[i % len(cluster_colors)])
            ax.fill(angles, values, alpha=0.25, color=cluster_colors[i % len(cluster_colors)])
            
            # Set category labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=8)
            
            # Remove radial labels and set grid alpha
            ax.set_yticklabels([])
            ax.grid(True, alpha=0.3)
            
            # Set title
            ax.set_title(f"Cluster {cluster_id} ({profile['Rule_Count']} rules)", 
                         fontsize=10, pad=15)
            
            # Add text annotation for TP rate and volume
            ax.text(0.5, -0.1, 
                   f"TP Rate: {profile['Avg_TP_Rate']:.1f}%\nVolume: {profile['Volume_Proportion']:.1f}%", 
                   ha='center', va='center', 
                   transform=ax.transAxes, fontsize=8)
        
        plt.tight_layout()
        plt.suptitle('Cluster Profiles (Multi-dimensional View)', fontsize=14, y=1.02)
        plt.savefig(f'{cluster_dir}/cluster_radar_profiles.png')
        plt.close()
        
        # 7. Create HTML report with cluster explanations
        html_report = f"""
        <html>
        <head>
            <title>Transaction Monitoring Rule Clustering Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #333366; }}
                .summary {{ background-color: #f8f8f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .high {{ color: green; font-weight: bold; }}
                .medium {{ color: orange; }}
                .low {{ color: red; }}
                .cluster-summary {{ margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Transaction Monitoring Rule Clustering Analysis</h1>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>This analysis has identified {optimal_clusters} natural groups of rules based on how they overlap in triggering alerts on the same KYC IDs. Understanding these clusters can help optimize the rule set by identifying redundancies and targeting improvements.</p>
                <p><strong>Key Findings:</strong></p>
                <ul>
        """
        
        # Add findings about cluster types
        high_perf_clusters = sum(1 for p in cluster_profiles if p['Cluster_Type'] == 'High-Performance')
        low_perf_clusters = sum(1 for p in cluster_profiles if p['Cluster_Type'] in ['Low-Performance', 'High-Volume, Low-Performance'])
        
        html_report += f"""
                    <li>{high_perf_clusters} high-performance clusters with average TP rates over 70%</li>
                    <li>{low_perf_clusters} low-performance clusters that may need optimization</li>
                </ul>
            </div>
            
            <h2>Cluster Overview</h2>
            <table>
                <tr>
                    <th>Cluster</th>
                    <th>Rules</th>
                    <th>TP Rate</th>
                    <th>Alert Volume</th>
                    <th>Dominant Pattern</th>
                    <th>Dominant Frequency</th>
                    <th>Cluster Type</th>
                    <th>Recommendation</th>
                </tr>
        """
        
        # Add rows for each cluster
        for profile in sorted(cluster_profiles, key=lambda x: x['Avg_TP_Rate'], reverse=True):
            # Determine color class based on TP rate
            if profile['Avg_TP_Rate'] >= 70:
                tp_class = "high"
            elif profile['Avg_TP_Rate'] >= 40:
                tp_class = "medium"
            else:
                tp_class = "low"
                
            html_report += f"""
                <tr>
                    <td>Cluster {profile['Cluster_ID']}</td>
                    <td>{profile['Rule_Count']}</td>
                    <td class="{tp_class}">{profile['Avg_TP_Rate']:.1f}%</td>
                    <td>{profile['Volume_Proportion']:.1f}% ({profile['Total_Alerts']} alerts)</td>
                    <td>{profile['Dominant_Pattern']}</td>
                    <td>{profile['Dominant_Frequency']}</td>
                    <td>{profile['Cluster_Type']}</td>
                    <td>{profile['Recommendation']}</td>
                </tr>
            """
        
        html_report += """
            </table>
            
            <h2>Detailed Cluster Analysis</h2>
        """
        
        # Add detailed section for each cluster
        for profile in sorted(cluster_profiles, key=lambda x: x['Cluster_ID']):
            cluster_rules_df = rule_clusters[rule_clusters['Cluster'] == profile['Cluster_ID'] - 1]
            
            html_report += f"""
            <div class="cluster-summary">
                <h3>Cluster {profile['Cluster_ID']}: {profile['Cluster_Type']}</h3>
                
                <p><strong>Characteristics:</strong></p>
                <ul>
                    <li>Contains {profile['Rule_Count']} rules</li>
                    <li>Average TP Rate: {profile['Avg_TP_Rate']:.1f}%</li>
                    <li>Generates {profile['Total_Alerts']} alerts ({profile['Volume_Proportion']:.1f}% of total)</li>
                    <li>Dominant Pattern: {profile['Dominant_Pattern']}</li>
                    <li>Dominant Frequency: {profile['Dominant_Frequency']}</li>
                </ul>
                
                <p><strong>Business Interpretation:</strong></p>
            """
            
            # Add business interpretation based on cluster type
            if profile['Cluster_Type'] == 'High-Performance':
                html_report += """
                <p>This cluster contains high-performing rules that are effectively identifying true positives. These rules are working well and should be maintained. The high true positive rate indicates these rules are capturing legitimate suspicious activity.</p>
                """
            elif profile['Cluster_Type'] == 'Medium-Performance':
                html_report += """
                <p>This cluster contains rules with moderate performance. While not optimal, these rules are still providing value. With tuning of thresholds, their performance could potentially be improved.</p>
                """
            elif profile['Cluster_Type'] == 'High-Volume, Low-Performance':
                html_report += """
                <p>This cluster generates a large volume of alerts but has a low true positive rate, indicating these rules may be contributing significantly to analyst workload without providing proportional value. These rules should be prioritized for review and potential threshold adjustments or removal.</p>
                """
            else:  # Low-Performance
                html_report += """
                <p>This cluster contains low-performing rules that are generating mostly false positives. These rules may be redundant or ineffective and should be evaluated for potential consolidation or removal.</p>
                """
            
            html_report += f"""
                <p><strong>Recommendation:</strong> {profile['Recommendation']}</p>
                
                <p><strong>Rules in this cluster:</strong></p>
                <table>
                    <tr>
                        <th>Rule</th>
                        <th>TP Rate</th>
                        <th>Alerts</th>
                        <th>Pattern</th>
                        <th>Frequency</th>
                    </tr>
            """
            
            # Add each rule in the cluster
            for _, rule in cluster_rules_df.sort_values('TP_Rate', ascending=False).iterrows():
                # Determine color class based on TP rate
                if rule['TP_Rate'] >= 70:
                    rule_tp_class = "high"
                elif rule['TP_Rate'] >= 40:
                    rule_tp_class = "medium"
                else:
                    rule_tp_class = "low"
                    
                html_report += f"""
                    <tr>
                        <td>{rule['Rule']}</td>
                        <td class="{rule_tp_class}">{rule['TP_Rate']:.1f}%</td>
                        <td>{int(rule['Total_Alerts'])}</td>
                        <td>{rule['Pattern']}</td>
                        <td>{rule['Frequency']}</td>
                    </tr>
                """
            
            html_report += """
                </table>
            </div>
            """
        
        # Close the HTML document
        html_report += """
        </body>
        </html>
        """
        
        # Write HTML report to file
        with open(f'{cluster_dir}/cluster_analysis_report.html', 'w') as f:
            f.write(html_report)
            
        print(f"Cluster analysis complete. HTML report saved to {cluster_dir}/cluster_analysis_report.html")
        
        # Return the clustering results and profiles
        return rule_clusters, cluster_profiles_df
    
    except Exception as e:
        print(f"Error in clustering: {e}")
        import traceback
        traceback.print_exc()
        # Create empty DataFrames to avoid further errors
        rule_clusters = pd.DataFrame(columns=['Rule', 'Cluster'])
        cluster_profiles_df = pd.DataFrame()
        return rule_clusters, cluster_profiles_df

# Execute the enhanced rule clustering
rule_clusters, cluster_profiles = perform_enhanced_rule_clustering(transaction_data, rule_performance, kyc_alerts)

# Print a summary of the cluster profiles
print("\nCluster profiles summary:")
if not cluster_profiles.empty:
    print(cluster_profiles[['Cluster_ID', 'Rule_Count', 'Avg_TP_Rate', 'Volume_Proportion', 'Cluster_Type']])
