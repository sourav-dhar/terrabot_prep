def create_rule_visualization(rule_code, merged_data):
    """Create a visualization for a specific rule, adapting to its pattern"""
    
    print(f"\nAnalyzing rule: {rule_code}")
    
    # Filter data for this rule
    rule_data = merged_data[merged_data['alert_rules'] == rule_code].copy()
    
    if rule_data.empty:
        print(f"No data found for rule {rule_code}")
        return None
    
    # Get rule information
    rule_pattern = rule_data['rule_pattern'].iloc[0] if 'rule_pattern' in rule_data.columns and not rule_data['rule_pattern'].empty else "Unknown"
    rule_frequency = rule_data['rule_frequency'].iloc[0] if 'rule_frequency' in rule_data.columns and not rule_data['rule_frequency'].empty else "Unknown"
    rule_threshold = rule_data['threshold_numeric'].iloc[0] if 'threshold_numeric' in rule_data.columns and not pd.isna(rule_data['threshold_numeric'].iloc[0]) else None
    rule_desc = rule_data['Rule description'].iloc[0] if 'Rule description' in rule_data.columns and not rule_data['Rule description'].empty else "Unknown"
    
    print(f"Pattern: {rule_pattern}, Frequency: {rule_frequency}")
    print(f"Threshold: {rule_threshold}")
    print(f"Description: {rule_desc}")
    
    # Check closed alerts for TP/FP analysis
    closed_rule_data = rule_data[rule_data['status'].isin(['Closed TP', 'Closed FP'])]
    if closed_rule_data.empty:
        print(f"No closed alerts (TP/FP) for rule {rule_code}. Using all data.")
        closed_rule_data = rule_data
    
    # Determine time periods based on rule frequency
    if rule_frequency == 'daily':
        closed_rule_data['time_period'] = closed_rule_data['transaction_date_time_local'].dt.date
        time_period_label = 'Transaction Date'
    elif rule_frequency == 'Weekly':
        closed_rule_data['time_period'] = closed_rule_data['transaction_date_time_local'].dt.strftime('%Y-W%U')
        time_period_label = 'Year-Week'
    elif rule_frequency == 'Monthly':
        closed_rule_data['time_period'] = closed_rule_data['transaction_date_time_local'].dt.strftime('%Y-%m')
        time_period_label = 'Year-Month'
    else:
        closed_rule_data['time_period'] = closed_rule_data['transaction_date_time_local'].dt.date
        time_period_label = 'Transaction Date'
    
    # Determine y-axis metric and aggregation based on rule pattern
    if isinstance(rule_pattern, str):
        if "Volume" in rule_pattern:
            # For Volume pattern, find appropriate amount field
            volume_fields = [col for col in closed_rule_data.columns if 'volume' in col.lower() or 'amount' in col.lower() or 'value' in col.lower()]
            if 'usd_value' in closed_rule_data.columns:
                y_metric = 'usd_value'
            elif volume_fields:
                y_metric = volume_fields[0]
            else:
                print("No suitable volume field found, using transaction count instead")
                y_metric = 'hub_transaction_id'
                
            y_agg = 'sum'
            y_label = f'Total {y_metric.replace("_", " ").title()}'
            
        elif "Velocity" in rule_pattern:
            # For Velocity pattern, count transactions
            y_metric = 'hub_transaction_id'
            y_agg = 'count'
            y_label = 'Transaction Count'
            
        elif "1 to Many" in rule_pattern:
            # For 1-to-Many pattern, count unique receivers
            y_metric = 'receiver_kyc_id_no'
            y_agg = 'nunique'
            y_label = 'Unique Receiver Count'
            
        elif "Many to 1" in rule_pattern:
            # For Many-to-1 pattern, count unique senders
            y_metric = 'sender_kyc_id_no'
            y_agg = 'nunique'
            y_label = 'Unique Sender Count'
            
        else:
            # Default to transaction count for unknown patterns
            y_metric = 'hub_transaction_id'
            y_agg = 'count'
            y_label = 'Transaction Count'
    else:
        # Default to transaction count for non-string patterns
        y_metric = 'hub_transaction_id'
        y_agg = 'count'
        y_label = 'Transaction Count'
    
    # Perform the aggregation with explicit column names
    print("Performing aggregation...")
    
    # Create a dictionary mapping columns to their aggregation functions
    agg_dict = {}
    
    # Add the metric aggregation
    if y_agg == 'nunique':
        agg_dict[y_metric] = pd.Series.nunique
    else:
        agg_dict[y_metric] = y_agg
        
    # Add transaction count aggregation (always a count)
    agg_dict['hub_transaction_id'] = 'count'
    
    # Add status aggregation (mode)
    agg_dict['status'] = lambda x: x.mode()[0] if not x.empty else 'Unknown'
    
    # Perform the aggregation
    aggregated_data = closed_rule_data.groupby(['alert_entity_id', 'time_period']).agg(agg_dict)
    
    # The aggregation result will have MultiIndex columns
    print("Column structure after aggregation:", aggregated_data.columns)
    
    # Flatten the columns if they're MultiIndex
    if isinstance(aggregated_data.columns, pd.MultiIndex):
        aggregated_data.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] for col in aggregated_data.columns]
    
    # Reset the index to make groupby columns into regular columns
    aggregated_data = aggregated_data.reset_index()
    print("Columns after reset_index:", aggregated_data.columns.tolist())
    
    # Create a column for the metric value with a clear name
    if f"{y_metric}_{y_agg}" in aggregated_data.columns:
        aggregated_data['metric_value'] = aggregated_data[f"{y_metric}_{y_agg}"]
    elif y_metric in aggregated_data.columns:
        aggregated_data['metric_value'] = aggregated_data[y_metric]
    else:
        # Find the most likely column for our metric
        potential_metric_cols = [col for col in aggregated_data.columns if y_metric in col]
        if potential_metric_cols:
            aggregated_data['metric_value'] = aggregated_data[potential_metric_cols[0]]
        else:
            print(f"Warning: Could not find metric column for {y_metric}. Using first numeric column.")
            numeric_cols = [col for col in aggregated_data.columns if aggregated_data[col].dtype.kind in 'if']
            if numeric_cols and numeric_cols[0] != 'alert_entity_id':
                aggregated_data['metric_value'] = aggregated_data[numeric_cols[0]]
            else:
                print("Error: No suitable numeric columns found for plotting.")
                return None
    
    # Create a size column - default to hub_transaction_id_count if available
    if 'hub_transaction_id_count' in aggregated_data.columns:
        aggregated_data['transaction_count'] = aggregated_data['hub_transaction_id_count']
    else:
        # Find the hub_transaction_id column with count aggregation
        transaction_count_col = [col for col in aggregated_data.columns if 'hub_transaction_id' in col and 'count' in col.lower()]
        if transaction_count_col:
            aggregated_data['transaction_count'] = aggregated_data[transaction_count_col[0]]
        else:
            # Create a dummy size column if needed
            print("Warning: No transaction count column found. Using constant size.")
            aggregated_data['transaction_count'] = 10  # Constant size as fallback
    
    print("Final columns for plotting:", aggregated_data.columns.tolist())
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Create scatter plot with correct column names
    scatter = sns.scatterplot(
        data=aggregated_data,
        x='time_period',
        y='metric_value',
        hue='status',
        size='transaction_count',
        sizes=(50, 300),
        alpha=0.7,
        palette={'Closed TP': 'green', 'Closed FP': 'red'}
    )
    
    # Add threshold line
    if rule_threshold is not None and not np.isnan(rule_threshold):
        plt.axhline(y=rule_threshold, color='blue', linestyle='--', label=f'Threshold: {rule_threshold}')
        
        # Add points above threshold text
        points_above = aggregated_data[aggregated_data['metric_value'] > rule_threshold].shape[0]
        total_points = aggregated_data.shape[0]
        percent_above = (points_above / total_points) * 100 if total_points > 0 else 0
        
        plt.text(
            0.05, 0.95, 
            f"Points above threshold: {points_above}/{total_points} ({percent_above:.1f}%)",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top'
        )
    
    # Add TP/FP counts
    tp_count = aggregated_data[aggregated_data['status'] == 'Closed TP'].shape[0]
    fp_count = aggregated_data[aggregated_data['status'] == 'Closed FP'].shape[0]
    
    plt.text(
        0.05, 0.90, 
        f"True Positives: {tp_count}, False Positives: {fp_count}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top'
    )
    
    # Set title and labels
    plt.title(f'Rule {rule_code}: {rule_desc}\n(Pattern: {rule_pattern}, Frequency: {rule_frequency})')
    plt.xlabel(time_period_label)
    plt.ylabel(y_label)
    
    # Format x-axis for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Handle x-axis for numerous time periods
    if len(aggregated_data['time_period'].unique()) > 20:
        # Show only a subset of time periods
        every_nth = max(1, len(aggregated_data['time_period'].unique()) // 20)
        for n, label in enumerate(plt.gca().xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
    
    # Add legend
    plt.legend(title='Status')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'rule_threshold_visualizations/rule_{rule_code}_analysis.png')
    plt.close()
    
    print(f"Analysis complete for rule {rule_code}. Visualization saved.")
    
    # Additional insights
    print("\nInsights:")
    
    # Calculate TP/FP rate
    total_classified = tp_count + fp_count
    tp_rate = (tp_count / total_classified) * 100 if total_classified > 0 else 0
    fp_rate = (fp_count / total_classified) * 100 if total_classified > 0 else 0
    
    print(f"- TP Rate: {tp_rate:.2f}%, FP Rate: {fp_rate:.2f}%")
    
    # Calculate average metric value for TP vs FP
    avg_metric_tp = aggregated_data[aggregated_data['status'] == 'Closed TP']['metric_value'].mean() if tp_count > 0 else np.nan
    avg_metric_fp = aggregated_data[aggregated_data['status'] == 'Closed FP']['metric_value'].mean() if fp_count > 0 else np.nan
    
    print(f"- Average {y_label} for TP: {avg_metric_tp:.2f}")
    print(f"- Average {y_label} for FP: {avg_metric_fp:.2f}")
    
    # Calculate threshold effectiveness
    if rule_threshold is not None and not np.isnan(rule_threshold):
        false_positives_above = aggregated_data[(aggregated_data['status'] == 'Closed FP') & 
                                             (aggregated_data['metric_value'] > rule_threshold)].shape[0]
        true_positives_below = aggregated_data[(aggregated_data['status'] == 'Closed TP') & 
                                             (aggregated_data['metric_value'] <= rule_threshold)].shape[0]
        
        print(f"- False positives above threshold: {false_positives_above}")
        print(f"- True positives below threshold: {true_positives_below}")
        
        # Suggest threshold adjustment if needed
        if false_positives_above > 0 or true_positives_below > 0:
            # Find potential optimal threshold
            
            # Only perform ROC analysis if we have both TP and FP
            if tp_count > 0 and fp_count > 0:
                try:
                    # Convert status to binary (1 for TP, 0 for FP)
                    y_true = (aggregated_data['status'] == 'Closed TP').astype(int)
                    y_scores = aggregated_data['metric_value']
                    
                    # Calculate ROC curve
                    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                    
                    # Find threshold with best balance (closest to top-left corner)
                    optimal_idx = np.argmin(np.sqrt((1-tpr)**2 + fpr**2))
                    optimal_threshold = thresholds[optimal_idx]
                    
                    print(f"- Suggested optimal threshold based on ROC analysis: {optimal_threshold:.2f}")
                except Exception as e:
                    print(f"  ROC analysis error: {e}")
                    
                    # Fallback to simple average if ROC analysis fails
                    if not np.isnan(avg_metric_tp) and not np.isnan(avg_metric_fp) and avg_metric_tp != avg_metric_fp:
                        suggested_threshold = (avg_metric_tp + avg_metric_fp) / 2
                        print(f"- Suggested threshold based on averages: {suggested_threshold:.2f}")
    
    return aggregated_data
