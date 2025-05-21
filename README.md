# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import os
import calendar
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from IPython.display import display, HTML

# Set visualization style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Create output directories
os.makedirs('visualizations', exist_ok=True)
os.makedirs('reports', exist_ok=True)

def load_transaction_data():
    """Load transaction data from Excel file"""
    # Read transaction data
    transaction_data = pd.read_excel('transaction_dummy_data_10k_final.xlsx', 
                                     sheet_name='transaction_dummy_data_10k')
    
    # Read rule descriptions
    rule_descriptions = pd.read_excel('transaction_dummy_data_10k_final.xlsx', 
                                     sheet_name='rule_description')
    
    # Convert dates to datetime
    date_columns = ['transaction_date_time_local', 'created_at', 'closed_at', 
                    'kyc_sender_create_date', 'kyc_receiver_create_date',
                    'dob_sender', 'dob_receiver', 'self_closure_date']
    
    for col in date_columns:
        if col in transaction_data.columns:
            transaction_data[col] = pd.to_datetime(transaction_data[col])
    
    print(f"Loaded {len(transaction_data)} transactions and {len(rule_descriptions)} rule descriptions")
    
    return transaction_data, rule_descriptions

def prepare_data_for_analysis(transaction_data):
    """Prepare transaction data for rule overlap analysis"""
    
    # Extract year and month from created_at
    if 'created_at' in transaction_data.columns:
        transaction_data['year'] = transaction_data['created_at'].dt.year
        transaction_data['month'] = transaction_data['created_at'].dt.month
        transaction_data['month_name'] = transaction_data['created_at'].dt.strftime('%b').str.upper()
        transaction_data['year_month'] = transaction_data['created_at'].dt.strftime('%Y-%m')
    else:
        # Fallback to transaction date if created_at is not available
        transaction_data['year'] = transaction_data['transaction_date_time_local'].dt.year
        transaction_data['month'] = transaction_data['transaction_date_time_local'].dt.month
        transaction_data['month_name'] = transaction_data['transaction_date_time_local'].dt.strftime('%b').str.upper()
        transaction_data['year_month'] = transaction_data['transaction_date_time_local'].dt.strftime('%Y-%m')
    
    # Determine the entity_id based on triggered_on
    transaction_data['entity_id'] = transaction_data.apply(
        lambda row: row['sender_kyc_id_no'] if row['triggered_on'] == 'sender' else row['receiver_kyc_id_no'],
        axis=1
    )
    
    print("Data prepared for analysis")
    print(f"Year-month combinations: {transaction_data['year_month'].unique()}")
    
    return transaction_data

def build_kyc_alert_mapping(transaction_data):
    """Build a mapping of KYC IDs to the rules they triggered"""
    
    # Initialize data structures
    # Global mapping (KYC ID -> set of rules)
    global_kyc_alerts = defaultdict(set)
    
    # Monthly mapping (Year-Month -> KYC ID -> set of rules)
    monthly_kyc_alerts = defaultdict(lambda: defaultdict(set))
    
    # Rule definition mapping
    rule_definitions = {}
    
    # Process each row
    for idx, row in transaction_data.iterrows():
        # Extract KYC ID based on triggered_on
        kyc_id = row['entity_id']
        rule = row['alert_rules']
        year_month = row['year_month']
        
        # Add to global mapping
        global_kyc_alerts[kyc_id].add(rule)
        
        # Add to monthly mapping
        monthly_kyc_alerts[year_month][kyc_id].add(rule)
        
        # Capture rule definition if available
        if 'Rule description' in row and rule not in rule_definitions:
            rule_definitions[rule] = row['Rule description']
    
    # Get unique rules
    all_rules = sorted(set(rule for rules in global_kyc_alerts.values() for rule in rules))
    
    # Get unique months sorted chronologically
    all_months = sorted(monthly_kyc_alerts.keys())
    
    print(f"Found {len(global_kyc_alerts)} unique KYC IDs across {len(all_months)} months")
    print(f"Found {len(all_rules)} unique rules")
    
    return global_kyc_alerts, monthly_kyc_alerts, all_rules, all_months, rule_definitions

def calculate_monthly_rule_stats(transaction_data, monthly_kyc_alerts, all_rules, all_months):
    """Calculate monthly statistics for each rule"""
    
    # Initialize results structure
    monthly_rule_stats = {}
    
    # Process each month
    for year_month in all_months:
        monthly_rule_stats[year_month] = {}
        
        # Get the month's data
        month_data = transaction_data[transaction_data['year_month'] == year_month]
        
        # Process each rule
        for rule in all_rules:
            # Filter data for this rule in this month
            rule_data = month_data[month_data['alert_rules'] == rule]
            
            # Count unique KYC IDs that triggered this rule this month
            rule_kycs = set()
            for kyc_id, rules in monthly_kyc_alerts[year_month].items():
                if rule in rules:
                    rule_kycs.add(kyc_id)
            
            kyc_count = len(rule_kycs)
            
            # Calculate TP/FP statistics if there are closed alerts
            closed_rule_data = rule_data[rule_data['status'].isin(['Closed TP', 'Closed FP'])]
            
            if not closed_rule_data.empty:
                tp_count = sum(closed_rule_data['status'] == 'Closed TP')
                fp_count = sum(closed_rule_data['status'] == 'Closed FP')
                total_closed = tp_count + fp_count
                
                tp_rate = (tp_count / total_closed * 100) if total_closed > 0 else 0
                fp_rate = (fp_count / total_closed * 100) if total_closed > 0 else 0
            else:
                tp_rate = 0
                fp_rate = 0
            
            # Store statistics
            monthly_rule_stats[year_month][rule] = {
                'kyc_count': kyc_count,
                'tp_rate': tp_rate,
                'fp_rate': fp_rate,
                'kyc_ids': rule_kycs
            }
    
    print("Monthly rule statistics calculated")
    return monthly_rule_stats

def calculate_rule_overlaps(monthly_kyc_alerts, all_rules, all_months):
    """Calculate rule overlap statistics for each month"""
    
    # Initialize results structure
    rule_overlaps = defaultdict(lambda: defaultdict(dict))
    
    # Process each month
    for year_month in all_months:
        month_kyc_alerts = monthly_kyc_alerts[year_month]
        
        # Process each rule pair
        for rule1 in all_rules:
            for rule2 in all_rules:
                if rule1 != rule2:
                    # Find KYCs that triggered rule1
                    rule1_kycs = set()
                    for kyc_id, rules in month_kyc_alerts.items():
                        if rule1 in rules:
                            rule1_kycs.add(kyc_id)
                    
                    # Find KYCs that triggered rule2
                    rule2_kycs = set()
                    for kyc_id, rules in month_kyc_alerts.items():
                        if rule2 in rules:
                            rule2_kycs.add(kyc_id)
                    
                    # Calculate overlap
                    overlap_kycs = rule1_kycs.intersection(rule2_kycs)
                    
                    # Directional overlap (percentage of rule1 KYCs that also triggered rule2)
                    if rule1_kycs:
                        directional_overlap_pct = len(overlap_kycs) / len(rule1_kycs) * 100
                    else:
                        directional_overlap_pct = 0
                    
                    # Jaccard similarity (intersection / union)
                    union_kycs = rule1_kycs.union(rule2_kycs)
                    if union_kycs:
                        jaccard_similarity = len(overlap_kycs) / len(union_kycs) * 100
                    else:
                        jaccard_similarity = 0
                    
                    # Store results
                    rule_overlaps[year_month][rule1][rule2] = {
                        'overlap_count': len(overlap_kycs),
                        'directional_overlap_pct': directional_overlap_pct,
                        'jaccard_similarity': jaccard_similarity,
                        'overlap_kycs': overlap_kycs
                    }
    
    print("Rule overlap statistics calculated")
    return rule_overlaps

def create_monthly_overlap_report(monthly_rule_stats, rule_overlaps, all_rules, all_months, rule_definitions):
    """Create a monthly overlap report as shown in the reference image"""
    
    # Initialize report data structure
    report_data = []
    
    # Map month numbers to abbreviations
    month_abbrs = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN',
                  7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}
    
    # Process each rule
    for rule in all_rules:
        # Get rule definition
        rule_def = rule_definitions.get(rule, "")
        
        row_data = {
            'rule': rule,
            'rule_definition': rule_def
        }
        
        # Add monthly statistics
        for year_month in all_months:
            # Extract month from year_month (format: YYYY-MM)
            month = int(year_month.split('-')[1])
            month_abbr = month_abbrs.get(month, str(month))
            
            # Get rule stats for this month
            rule_stats = monthly_rule_stats[year_month].get(rule, {})
            
            kyc_count = rule_stats.get('kyc_count', 0)
            tp_rate = rule_stats.get('tp_rate', 0)
            fp_rate = rule_stats.get('fp_rate', 0)
            
            # Add to row data
            row_data[f'no_of_kyc_alerted_{month_abbr}'] = kyc_count
            row_data[f'%_CLOSED_TP_{month_abbr}'] = tp_rate
            row_data[f'%_CLOSED_FP_{month_abbr}'] = fp_rate
        
        # Add rule overlap statistics
        # For demonstration, we'll use the last month's data for overlap
        last_month = all_months[-1] if all_months else None
        
        if last_month:
            # Add overlap with each other rule
            for other_rule in all_rules:
                if other_rule != rule:
                    overlap_stats = rule_overlaps[last_month][rule].get(other_rule, {})
                    directional_overlap_pct = overlap_stats.get('directional_overlap_pct', 0)
                    
                    row_data[f'{other_rule}_% OVERLAP'] = directional_overlap_pct
        
        report_data.append(row_data)
    
    # Create DataFrame from report data
    report_df = pd.DataFrame(report_data)
    
    print("Monthly overlap report created")
    return report_df

def calculate_best_thresholds(transaction_data, monthly_rule_stats, all_rules, all_months):
    """Calculate optimal thresholds for each rule based on TP/FP rates"""
    
    threshold_recommendations = {}
    
    for rule in all_rules:
        # Get all KYCs that triggered this rule across months
        rule_kycs = set()
        for year_month in all_months:
            rule_stats = monthly_rule_stats[year_month].get(rule, {})
            kyc_ids = rule_stats.get('kyc_ids', set())
            rule_kycs.update(kyc_ids)
        
        # Get all transactions for these KYCs
        rule_transactions = transaction_data[
            (transaction_data['entity_id'].isin(rule_kycs)) & 
            (transaction_data['alert_rules'] == rule)
        ]
        
        # If we have enough data with known status
        closed_transactions = rule_transactions[rule_transactions['status'].isin(['Closed TP', 'Closed FP'])]
        
        if len(closed_transactions) >= 10:  # Require at least 10 data points
            # Try to find a metric to optimize
            # First check if we have transaction amounts
            if 'usd_value' in closed_transactions.columns:
                metric = 'usd_value'
            elif any(col for col in closed_transactions.columns if 'amount' in col.lower()):
                amount_cols = [col for col in closed_transactions.columns if 'amount' in col.lower()]
                metric = amount_cols[0]
            else:
                # No clear metric, skip threshold optimization
                continue
            
            # Extract TP and FP transaction amounts
            tp_values = closed_transactions[closed_transactions['status'] == 'Closed TP'][metric].dropna()
            fp_values = closed_transactions[closed_transactions['status'] == 'Closed FP'][metric].dropna()
            
            if len(tp_values) > 0 and len(fp_values) > 0:
                # Calculate basic statistics
                tp_mean = tp_values.mean()
                fp_mean = fp_values.mean()
                tp_median = tp_values.median()
                fp_median = fp_values.median()
                
                # Find a threshold that maximizes separation
                min_val = min(tp_values.min(), fp_values.min())
                max_val = max(tp_values.max(), fp_values.max())
                
                best_threshold = None
                best_score = -float('inf')
                
                # Try different thresholds
                thresholds = np.linspace(min_val, max_val, 100)
                for threshold in thresholds:
                    # Calculate TP rate and FP rate at this threshold
                    tp_above = (tp_values > threshold).mean()
                    fp_above = (fp_values > threshold).mean()
                    
                    # Score is TP rate - FP rate (maximize)
                    score = tp_above - fp_above
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                
                # Store recommendation
                threshold_recommendations[rule] = {
                    'best_threshold': best_threshold,
                    'tp_mean': tp_mean,
                    'fp_mean': fp_mean,
                    'tp_median': tp_median,
                    'fp_median': fp_median,
                    'score': best_score
                }
    
    print(f"Calculated threshold recommendations for {len(threshold_recommendations)} rules")
    return threshold_recommendations

def generate_rule_insights(transaction_data, monthly_rule_stats, rule_overlaps, all_rules, all_months):
    """Generate insights for each rule"""
    
    rule_insights = {}
    
    for rule in all_rules:
        insights = []
        
        # Trend analysis
        kyc_counts = []
        tp_rates = []
        fp_rates = []
        
        for year_month in all_months:
            rule_stats = monthly_rule_stats[year_month].get(rule, {})
            kyc_counts.append(rule_stats.get('kyc_count', 0))
            tp_rates.append(rule_stats.get('tp_rate', 0))
            fp_rates.append(rule_stats.get('fp_rate', 0))
        
        # KYC count trend
        if len(kyc_counts) >= 2:
            kyc_trend = kyc_counts[-1] - kyc_counts[0]
            if kyc_trend > 0:
                insights.append(f"KYC alerts increased by {kyc_trend} from first to last month")
            elif kyc_trend < 0:
                insights.append(f"KYC alerts decreased by {abs(kyc_trend)} from first to last month")
        
        # TP/FP rate trend
        if len(tp_rates) >= 2 and tp_rates[0] > 0 and tp_rates[-1] > 0:
            tp_trend = tp_rates[-1] - tp_rates[0]
            if abs(tp_trend) >= 5:  # Only report significant changes
                if tp_trend > 0:
                    insights.append(f"TP rate improved by {tp_trend:.1f}% from first to last month")
                else:
                    insights.append(f"TP rate declined by {abs(tp_trend):.1f}% from first to last month")
        
        # Overlap insights
        if all_months:
            last_month = all_months[-1]
            
            # Find rules with high overlap
            high_overlaps = []
            for other_rule in all_rules:
                if other_rule != rule:
                    overlap_stats = rule_overlaps[last_month][rule].get(other_rule, {})
                    directional_overlap_pct = overlap_stats.get('directional_overlap_pct', 0)
                    
                    if directional_overlap_pct >= 50:  # High overlap threshold
                        high_overlaps.append((other_rule, directional_overlap_pct))
            
            # Sort by overlap percentage
            high_overlaps.sort(key=lambda x: x[1], reverse=True)
            
            if high_overlaps:
                top_overlaps = high_overlaps[:3]  # Top 3 overlapping rules
                overlap_str = ", ".join([f"{rule} ({pct:.1f}%)" for rule, pct in top_overlaps])
                insights.append(f"High overlap with: {overlap_str}")
        
        # Calculate effectiveness
        total_kyc_count = sum(kyc_counts)
        avg_tp_rate = np.mean([r for r in tp_rates if r > 0]) if any(r > 0 for r in tp_rates) else 0
        
        if total_kyc_count > 0 and avg_tp_rate > 0:
            if avg_tp_rate >= 70:
                insights.append("Highly effective rule with good TP rate")
            elif avg_tp_rate <= 30:
                insights.append("Low effectiveness rule with poor TP rate")
        
        rule_insights[rule] = insights
    
    print("Generated insights for all rules")
    return rule_insights

def create_enhanced_report(report_df, monthly_rule_stats, rule_overlaps, threshold_recommendations, rule_insights, all_rules, all_months):
    """Create an enhanced report with additional analysis"""
    
    # Create report directory
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    
    # Save basic report to CSV
    report_csv_path = os.path.join(report_dir, "monthly_rule_overlap_report.csv")
    report_df.to_csv(report_csv_path, index=False)
    print(f"Basic report saved to {report_csv_path}")
    
    # Create enhanced Excel report
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
    
    wb = Workbook()
    
    # Monthly Overlap Sheet
    ws_monthly = wb.active
    ws_monthly.title = "Monthly Rule Overlap"
    
    # Write headers
    headers = list(report_df.columns)
    for col_idx, header in enumerate(headers, 1):
        cell = ws_monthly.cell(row=1, column=col_idx)
        cell.value = header
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="C9DAF8", end_color="C9DAF8", fill_type="solid")
        cell.alignment = Alignment(horizontal='center')
    
    # Write data
    for row_idx, row in enumerate(dataframe_to_rows(report_df, index=False, header=False), 2):
        for col_idx, value in enumerate(row, 1):
            cell = ws_monthly.cell(row=row_idx, column=col_idx)
            cell.value = value
            
            # Format percentage cells
            if headers[col_idx-1].startswith('%') or 'OVERLAP' in headers[col_idx-1]:
                if isinstance(value, (int, float)):
                    cell.value = value / 100  # Convert to decimal for Excel percentage format
                    cell.number_format = '0.00%'
    
    # Threshold Recommendations Sheet
    ws_thresholds = wb.create_sheet(title="Threshold Recommendations")
    
    # Write headers
    threshold_headers = ["Rule", "Recommended Threshold", "TP Mean", "FP Mean", "TP/FP Separation Score"]
    for col_idx, header in enumerate(threshold_headers, 1):
        cell = ws_thresholds.cell(row=1, column=col_idx)
        cell.value = header
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="C9DAF8", end_color="C9DAF8", fill_type="solid")
        cell.alignment = Alignment(horizontal='center')
    
    # Write threshold data
    row_idx = 2
    for rule, data in threshold_recommendations.items():
        ws_thresholds.cell(row=row_idx, column=1).value = rule
        ws_thresholds.cell(row=row_idx, column=2).value = data['best_threshold']
        ws_thresholds.cell(row=row_idx, column=3).value = data['tp_mean']
        ws_thresholds.cell(row=row_idx, column=4).value = data['fp_mean']
        ws_thresholds.cell(row=row_idx, column=5).value = data['score']
        row_idx += 1
    
    # Rule Insights Sheet
    ws_insights = wb.create_sheet(title="Rule Insights")
    
    # Write headers
    insight_headers = ["Rule", "Insights"]
    for col_idx, header in enumerate(insight_headers, 1):
        cell = ws_insights.cell(row=1, column=col_idx)
        cell.value = header
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="C9DAF8", end_color="C9DAF8", fill_type="solid")
        cell.alignment = Alignment(horizontal='center')
    
    # Write insight data
    row_idx = 2
    for rule, insights in rule_insights.items():
        ws_insights.cell(row=row_idx, column=1).value = rule
        ws_insights.cell(row=row_idx, column=2).value = "\n".join(insights) if insights else "No significant insights"
        ws_insights.cell(row=row_idx, column=2).alignment = Alignment(wrapText=True)
        row_idx += 1
    
    # Adjust column widths
    for sheet in wb.worksheets:
        for column in sheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            
            for cell in column:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            
            adjusted_width = max(max_length + 2, 12)
            sheet.column_dimensions[column_letter].width = adjusted_width
    
    # Save Excel file
    excel_path = os.path.join(report_dir, "enhanced_rule_overlap_report.xlsx")
    wb.save(excel_path)
    print(f"Enhanced report saved to {excel_path}")
    
    return report_csv_path, excel_path

def visualize_rule_overlaps(monthly_rule_stats, rule_overlaps, all_rules, all_months):
    """Create visualizations for rule overlaps"""
    
    # Create visualization directory
    viz_dir = "visualizations/rule_overlaps"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Select the most recent month for visualization
    recent_month = all_months[-1] if all_months else None
    if not recent_month:
        print("No months available for visualization")
        return
    
    # 1. Rule Overlap Heatmap
    plt.figure(figsize=(12, 10))
    
    # Create overlap matrix
    overlap_matrix = np.zeros((len(all_rules), len(all_rules)))
    
    for i, rule1 in enumerate(all_rules):
        for j, rule2 in enumerate(all_rules):
            if rule1 != rule2:
                overlap_stats = rule_overlaps[recent_month][rule1].get(rule2, {})
                directional_overlap_pct = overlap_stats.get('directional_overlap_pct', 0)
                overlap_matrix[i, j] = directional_overlap_pct
    
    # Create custom colormap: white for 0, gradual blue for higher values
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', '#9CC3E6', '#2E75B6', '#1F4E79'])
    
    # Plot heatmap
    sns.heatmap(overlap_matrix, xticklabels=all_rules, yticklabels=all_rules,
                cmap=cmap, vmin=0, vmax=100, annot=True, fmt=".1f", 
                linewidth=0.5, linecolor='lightgray')
    
    plt.title(f'Rule Overlap Percentages - {recent_month}')
    plt.xlabel('Rule (% of row rule KYCs also in column rule)')
    plt.ylabel('Rule')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"rule_overlap_heatmap_{recent_month.replace('-', '_')}.png"), dpi=300)
    plt.close()
    
    # 2. Rule Network Graph
    plt.figure(figsize=(14, 12))
    
    G = nx.Graph()
    
    # Add nodes (rules)
    for rule in all_rules:
        # Get KYC count for node size
        rule_stats = monthly_rule_stats[recent_month].get(rule, {})
        kyc_count = rule_stats.get('kyc_count', 0)
        tp_rate = rule_stats.get('tp_rate', 0)
        
        # Skip rules with no activity this month
        if kyc_count > 0:
            G.add_node(rule, kyc_count=kyc_count, tp_rate=tp_rate)
    
    # Add edges (overlaps)
    for rule1 in all_rules:
        if rule1 in G.nodes():
            for rule2 in all_rules:
                if rule2 in G.nodes() and rule1 != rule2:
                    overlap_stats = rule_overlaps[recent_month][rule1].get(rule2, {})
                    jaccard_similarity = overlap_stats.get('jaccard_similarity', 0)
                    
                    # Only add edges for significant overlaps
                    if jaccard_similarity >= 10:  # At least 10% Jaccard similarity
                        G.add_edge(rule1, rule2, weight=jaccard_similarity)
    
    # Skip visualization if network is empty
    if not G.nodes():
        print("No rules with activity this month for network visualization")
        return
    
    # Visualization parameters
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Node colors based on TP rate
    node_colors = [G.nodes[rule]['tp_rate'] for rule in G.nodes()]
    color_map = plt.cm.RdYlGn  # Red-Yellow-Green colormap
    
    # Node sizes based on KYC count
    node_sizes = [200 + G.nodes[rule]['kyc_count'] * 5 for rule in G.nodes()]
    
    # Edge weights based on Jaccard similarity
    edge_weights = [G[u][v]['weight'] / 10 for u, v in G.edges()]
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, cmap=color_map)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=9)
    
    # Add colorbar for TP rate
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('True Positive Rate (%)')
    
    plt.title(f'Rule Overlap Network - {recent_month}\nNode size: KYC count, Edge width: Jaccard similarity, Color: TP rate')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"rule_overlap_network_{recent_month.replace('-', '_')}.png"), dpi=300)
    plt.close()
    
    # 3. Monthly Trend Charts for Top Rules
    # Identify top rules by KYC count in the most recent month
    recent_kyc_counts = [(rule, monthly_rule_stats[recent_month].get(rule, {}).get('kyc_count', 0)) 
                         for rule in all_rules]
    top_rules = [rule for rule, count in sorted(recent_kyc_counts, key=lambda x: x[1], reverse=True)[:5]]
    
    plt.figure(figsize=(14, 8))
    
    # Plot KYC count trend
    for rule in top_rules:
        kyc_counts = [monthly_rule_stats[month].get(rule, {}).get('kyc_count', 0) for month in all_months]
        plt.plot(all_months, kyc_counts, 'o-', label=rule, linewidth=2, markersize=8)
    
    plt.title('Monthly KYC Alert Trend for Top Rules')
    plt.xlabel('Month')
    plt.ylabel('Number of KYC Alerts')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "top_rules_kyc_trend.png"), dpi=300)
    plt.close()
    
    # 4. TP Rate Trend
    plt.figure(figsize=(14, 8))
    
    for rule in top_rules:
        tp_rates = [monthly_rule_stats[month].get(rule, {}).get('tp_rate', 0) for month in all_months]
        plt.plot(all_months, tp_rates, 'o-', label=rule, linewidth=2, markersize=8)
    
    plt.title('Monthly True Positive Rate Trend for Top Rules')
    plt.xlabel('Month')
    plt.ylabel('True Positive Rate (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "top_rules_tp_trend.png"), dpi=300)
    plt.close()
    
    print(f"Rule overlap visualizations saved to {viz_dir}")

def analyze_rule_overlap_for_specific_rule(transaction_data, monthly_kyc_alerts, rule_overlaps, target_rule, month=None):
    """
    Perform detailed overlap analysis for a specific rule
    
    This is an enhanced version of the provided analyze_specific_rule_overlap function
    """
    # Determine month to analyze
    if month is None:
        # Use the most recent month
        all_months = sorted(monthly_kyc_alerts.keys())
        if not all_months:
            print("No months available for analysis")
            return None
        month = all_months[-1]
    
    print(f"\nDetailed overlap analysis for rule {target_rule} in month {month}")
    
    # Get all KYCs that triggered this rule in this month
    target_kycs = set()
    for kyc_id, rules in monthly_kyc_alerts[month].items():
        if target_rule in rules:
            target_kycs.add(kyc_id)
    
    total_kycs = len(target_kycs)
    print(f"Total KYC IDs that triggered {target_rule} in {month}: {total_kycs}")
    
    if total_kycs == 0:
        print(f"No KYCs triggered rule {target_rule} in {month}")
        return None
    
    # Get all rules in this month
    all_rules = set()
    for kyc_id, rules in monthly_kyc_alerts[month].items():
        all_rules.update(rules)
    all_rules = sorted(all_rules)
    
    # Count co-occurrences with other rules
    rule_overlaps_count = {}
    for other_rule in all_rules:
        if other_rule != target_rule:
            overlap_stats = rule_overlaps[month][target_rule].get(other_rule, {})
            overlap_count = overlap_stats.get('overlap_count', 0)
            directional_pct = overlap_stats.get('directional_overlap_pct', 0)
            jaccard_pct = overlap_stats.get('jaccard_similarity', 0)
            
            rule_overlaps_count[other_rule] = {
                'overlap_count': overlap_count,
                'directional_pct': directional_pct,
                'jaccard_pct': jaccard_pct
            }
    
    # Sort by overlap count
    sorted_overlaps = sorted(rule_overlaps_count.items(), 
                             key=lambda x: x[1]['overlap_count'], 
                             reverse=True)
    
    # Display top overlapping rules
    print("\nTop rules that co-occur with {}:".format(target_rule))
    for rule, stats in sorted_overlaps[:10]:  # Show top 10
        print(f"  {rule}: {stats['overlap_count']} KYCs ({stats['directional_pct']:.2f}% of {target_rule} KYCs)")
    
    # Create visualization directory
    viz_dir = "visualizations/rule_specific"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Visualize overlap with top co-occurring rules
    if sorted_overlaps:
        plt.figure(figsize=(12, 8))
        
        # Get top overlapping rules
        top_n = min(10, len(sorted_overlaps))
        top_rules = [rule for rule, _ in sorted_overlaps[:top_n]]
        overlap_counts = [stats['overlap_count'] for rule, stats in sorted_overlaps[:top_n]]
        directional_pcts = [stats['directional_pct'] for rule, stats in sorted_overlaps[:top_n]]
        
        # Sort for better visualization
        indices = np.argsort(overlap_counts)
        top_rules = [top_rules[i] for i in indices]
        overlap_counts = [overlap_counts[i] for i in indices]
        directional_pcts = [directional_pcts[i] for i in indices]
        
        # Create bar chart
        y_pos = np.arange(len(top_rules))
        
        # Primary axis: overlap counts
        plt.barh(y_pos, overlap_counts, color='#1F4E79', alpha=0.8)
        plt.yticks(y_pos, top_rules)
        plt.xlabel('Number of Overlapping KYCs')
        
        # Secondary axis: percentage overlap
        ax2 = plt.twinx()
        for i, (count, pct) in enumerate(zip(overlap_counts, directional_pcts)):
            ax2.annotate(f"{pct:.1f}%", 
                         xy=(count, i), 
                         xytext=(5, 0), 
                         textcoords="offset points", 
                         va='center', 
                         fontweight='bold', 
                         color='#C65911')
        
        ax2.set_ylim(plt.ylim())
        ax2.set_ylabel('Percentage of Target Rule KYCs')
        
        plt.title(f'Rules Co-occurring with {target_rule} in {month}')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{target_rule}_overlap_{month.replace('-', '_')}.png"), dpi=300)
        plt.close()
    
    # Calculate KYC distribution (how many rules per KYC)
    rule_counts_per_kyc = {}
    for kyc_id in target_kycs:
        rule_count = len(monthly_kyc_alerts[month].get(kyc_id, set()))
        rule_counts_per_kyc[kyc_id] = rule_count
    
    rule_count_distribution = pd.Series(list(rule_counts_per_kyc.values())).value_counts().sort_index()
    
    print("\nDistribution of number of rules per KYC ID:")
    print(rule_count_distribution)
    
    # Visualize rule count distribution
    plt.figure(figsize=(10, 6))
    rule_count_distribution.plot(kind='bar', color='#4472C4')
    plt.title(f'Number of Rules Triggered per KYC ID for {target_rule} in {month}')
    plt.xlabel('Number of Rules')
    plt.ylabel('Count of KYC IDs')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{target_rule}_rule_counts_{month.replace('-', '_')}.png"), dpi=300)
    plt.close()
    
    # Analyze TP/FP rates for overlapping KYCs
    # First, get a mapping of KYC to status
    kyc_status = {}
    for kyc_id in target_kycs:
        # Get all alerts for this KYC and this rule
        kyc_alerts = transaction_data[
            (transaction_data['entity_id'] == kyc_id) & 
            (transaction_data['alert_rules'] == target_rule) &
            (transaction_data['year_month'] == month)
        ]
        
        # If we have closed alerts, get the most common status
        closed_alerts = kyc_alerts[kyc_alerts['status'].isin(['Closed TP', 'Closed FP'])]
        if not closed_alerts.empty:
            most_common_status = closed_alerts['status'].mode()[0]
            kyc_status[kyc_id] = most_common_status
    
    # Check if the overlaps are more common in TP or FP cases
    for rule, stats in sorted_overlaps[:5]:  # Analyze top 5
        overlap_kycs = rule_overlaps[month][target_rule][rule]['overlap_kycs']
        
        tp_overlaps = sum(1 for kyc in overlap_kycs if kyc_status.get(kyc) == 'Closed TP')
        fp_overlaps = sum(1 for kyc in overlap_kycs if kyc_status.get(kyc) == 'Closed FP')
        
        total_closed = tp_overlaps + fp_overlaps
        if total_closed > 0:
            tp_pct = tp_overlaps / total_closed * 100
            fp_pct = fp_overlaps / total_closed * 100
            
            print(f"\nOverlap of {target_rule} with {rule}:")
            print(f"  True Positives: {tp_overlaps} ({tp_pct:.1f}%)")
            print(f"  False Positives: {fp_overlaps} ({fp_pct:.1f}%)")
    
    return target_kycs, rule_overlaps_count, rule_count_distribution

def main():
    """Main function to execute the rule overlap analysis"""
    
    # 1. Load and prepare data
    transaction_data, rule_descriptions = load_transaction_data()
    transaction_data = prepare_data_for_analysis(transaction_data)
    
    # 2. Build KYC to alert rule mappings
    global_kyc_alerts, monthly_kyc_alerts, all_rules, all_months, rule_definitions = build_kyc_alert_mapping(transaction_data)
    
    # 3. Calculate monthly statistics for each rule
    monthly_rule_stats = calculate_monthly_rule_stats(transaction_data, monthly_kyc_alerts, all_rules, all_months)
    
    # 4. Calculate rule overlaps
    rule_overlaps = calculate_rule_overlaps(monthly_kyc_alerts, all_rules, all_months)
    
    # 5. Create monthly overlap report
    report_df = create_monthly_overlap_report(monthly_rule_stats, rule_overlaps, all_rules, all_months, rule_definitions)
    
    # 6. Calculate best thresholds
    threshold_recommendations = calculate_best_thresholds(transaction_data, monthly_rule_stats, all_rules, all_months)
    
    # 7. Generate rule insights
    rule_insights = generate_rule_insights(transaction_data, monthly_rule_stats, rule_overlaps, all_rules, all_months)
    
    # 8. Create enhanced report
    report_csv_path, excel_path = create_enhanced_report(
        report_df, monthly_rule_stats, rule_overlaps, 
        threshold_recommendations, rule_insights, 
        all_rules, all_months
    )
    
    # 9. Visualize rule overlaps
    visualize_rule_overlaps(monthly_rule_stats, rule_overlaps, all_rules, all_months)
    
    # 10. Detailed analysis for specific rules
    # Example: Analyze TRP_0001
    target_rule = "TRP_0001"
    analyze_rule_overlap_for_specific_rule(transaction_data, monthly_kyc_alerts, rule_overlaps, target_rule)
    
    print("\nAnalysis complete!")
    print(f"Basic report: {report_csv_path}")
    print(f"Enhanced report: {excel_path}")
    print("Visualizations saved to the 'visualizations' directory")

if __name__ == "__main__":
    main()
