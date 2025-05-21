# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import os
import calendar
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Create output directories
os.makedirs('visualizations', exist_ok=True)
os.makedirs('reports', exist_ok=True)

def load_transaction_data():
    """Load transaction data from Excel file and join with rule descriptions"""
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
    
    # Join rule descriptions with transaction data
    transaction_data = transaction_data.merge(
        rule_descriptions[['Rule no.', 'Rule description']],
        left_on='alert_rules',
        right_on='Rule no.',
        how='left'
    )
    
    print(f"Loaded {len(transaction_data)} transactions and {len(rule_descriptions)} rule descriptions")
    print(f"Number of unique rules: {transaction_data['alert_rules'].nunique()}")
    
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
    
    # Add a unique identifier for each alert (important for accurate counting)
    transaction_data['alert_id'] = range(len(transaction_data))
    
    print("Data prepared for analysis")
    print(f"Year-month combinations: {transaction_data['year_month'].unique()}")
    
    return transaction_data

def build_monthly_alert_counts(transaction_data):
    """Build accurate monthly alert counts for each rule"""
    
    # Get all unique rules and months
    all_rules = sorted(transaction_data['alert_rules'].unique())
    all_months = sorted(transaction_data['year_month'].unique())
    
    # Create rule definitions map directly from transaction_data
    rule_definitions = {}
    for rule in all_rules:
        # Get the rule description from any row with this rule
        rule_rows = transaction_data[transaction_data['alert_rules'] == rule]
        if not rule_rows.empty and 'Rule description' in rule_rows.columns:
            description = rule_rows['Rule description'].iloc[0]
            if pd.notna(description):
                rule_definitions[rule] = description
            else:
                rule_definitions[rule] = ""
        else:
            rule_definitions[rule] = ""
    
    # Initialize results structure
    monthly_stats = {}
    
    # Process each month
    for month in all_months:
        month_data = transaction_data[transaction_data['year_month'] == month]
        
        # Convert month number to name
        month_abbr = month.split('-')[1].zfill(2)
        month_name = {'01': 'JAN', '02': 'FEB', '03': 'MAR', '04': 'APR', '05': 'MAY', 
                      '06': 'JUN', '07': 'JUL', '08': 'AUG', '09': 'SEP', '10': 'OCT', 
                      '11': 'NOV', '12': 'DEC'}.get(month_abbr, month_abbr)
        
        monthly_stats[month_name] = {}
        
        # Process each rule
        for rule in all_rules:
            # Filter data for this rule in this month
            rule_data = month_data[month_data['alert_rules'] == rule]
            
            # Count total alerts - each row is an alert
            alert_count = len(rule_data)
            
            # Count unique KYC IDs that triggered this rule
            # Use alert_entity_id as specified
            if 'alert_entity_id' in rule_data.columns:
                kyc_count = rule_data['alert_entity_id'].nunique()
                kyc_ids = set(rule_data['alert_entity_id'].unique())
            else:
                # Fallback in case alert_entity_id doesn't exist
                entity_id_field = 'entity_id' if 'entity_id' in rule_data.columns else 'sender_kyc_id_no'
                kyc_count = rule_data[entity_id_field].nunique()
                kyc_ids = set(rule_data[entity_id_field].unique())
            
            # Calculate TP/FP statistics
            closed_rule_data = rule_data[rule_data['status'].isin(['Closed TP', 'Closed FP'])]
            
            if not closed_rule_data.empty:
                tp_count = sum(closed_rule_data['status'] == 'Closed TP')
                fp_count = sum(closed_rule_data['status'] == 'Closed FP')
                total_closed = tp_count + fp_count
                
                tp_pct = round(tp_count / total_closed * 100, 1) if total_closed > 0 else 0
                fp_pct = round(fp_count / total_closed * 100, 1) if total_closed > 0 else 0
            else:
                tp_pct = 0
                fp_pct = 0
            
            # Store statistics
            monthly_stats[month_name][rule] = {
                'alert_count': alert_count,
                'kyc_count': kyc_count,
                'tp_pct': tp_pct,
                'fp_pct': fp_pct,
                # Store unique KYC IDs for overlap calculation
                'kyc_ids': kyc_ids
            }
    
    print("Monthly alert counts calculated accurately")
    month_names = list(monthly_stats.keys())
    
    return monthly_stats, all_rules, month_names, rule_definitions

def calculate_rule_overlap(monthly_stats, all_rules, all_months):
    """Calculate rule overlap percentages accurately"""
    
    # Initialize overlap data structure
    rule_overlaps = {}
    
    # For each month
    for month in all_months:
        rule_overlaps[month] = {}
        
        # For each rule pair
        for rule1 in all_rules:
            rule_overlaps[month][rule1] = {}
            
            # Get KYCs for rule1
            rule1_kycs = monthly_stats[month][rule1]['kyc_ids']
            rule1_kyc_count = len(rule1_kycs)
            
            # Skip if no KYCs for rule1
            if rule1_kyc_count == 0:
                continue
            
            # Calculate total overlap across all other rules
            all_overlapping_kycs = set()
            for rule2 in all_rules:
                if rule1 != rule2:
                    # Get KYCs for rule2
                    rule2_kycs = monthly_stats[month][rule2]['kyc_ids']
                    
                    # Find overlap (intersection)
                    overlap_kycs = rule1_kycs.intersection(rule2_kycs)
                    all_overlapping_kycs.update(overlap_kycs)
                    
                    overlap_count = len(overlap_kycs)
                    
                    # Calculate overlap percentage (what percentage of rule1 KYCs also triggered rule2)
                    overlap_pct = round(overlap_count / rule1_kyc_count * 100, 1) if rule1_kyc_count > 0 else 0
                    
                    # Store overlap stats
                    rule_overlaps[month][rule1][rule2] = {
                        'overlap_count': overlap_count,
                        'overlap_pct': overlap_pct,
                        'overlap_kycs': overlap_kycs
                    }
            
            # Calculate total overlap percentage
            total_overlap_count = len(all_overlapping_kycs)
            total_overlap_pct = round(total_overlap_count / rule1_kyc_count * 100, 1) if rule1_kyc_count > 0 else 0
            
            # Store total overlap
            rule_overlaps[month][rule1]['TOTAL_OVERLAP'] = {
                'overlap_count': total_overlap_count,
                'overlap_pct': total_overlap_pct,
                'overlap_kycs': all_overlapping_kycs
            }
    
    print("Rule overlaps calculated accurately")
    return rule_overlaps

def create_monthly_overlap_report(monthly_stats, rule_overlaps, all_rules, all_months, rule_definitions):
    """Create a monthly overlap report with accurate counts and percentages"""
    
    # Initialize the dataframe that will hold our report
    report_data = []
    
    # For each rule
    for rule in all_rules:
        row = {
            'rule': rule,
            'rule_definition': rule_definitions.get(rule, "")  # Make sure this gets the definition
        }
        
        # Add total overlap percentage for the most recent month
        last_month = all_months[-1]  # Get the last month
        total_overlap = rule_overlaps[last_month][rule].get('TOTAL_OVERLAP', {'overlap_pct': 0})
        row['TOTAL_% OVERLAP'] = total_overlap['overlap_pct']
        
        # Add monthly statistics
        for month in all_months:
            month_data = monthly_stats[month][rule]
            
            row[f'no_of_kyc_alerted_{month}'] = month_data['kyc_count']
            row[f'%_CLOSED_TP_{month}'] = month_data['tp_pct']
            row[f'%_CLOSED_FP_{month}'] = month_data['fp_pct']
        
        # Add overlap percentages for the most recent month
        for other_rule in all_rules:
            if rule != other_rule:
                if other_rule in rule_overlaps[last_month][rule]:
                    overlap_pct = rule_overlaps[last_month][rule][other_rule]['overlap_pct']
                    row[f'{other_rule}_% OVERLAP'] = overlap_pct
                else:
                    row[f'{other_rule}_% OVERLAP'] = 0
        
        report_data.append(row)
    
    # Create dataframe
    report_df = pd.DataFrame(report_data)
    
    # Reorder columns to put TOTAL_% OVERLAP right after rule_definition
    cols = list(report_df.columns)
    cols.remove('TOTAL_% OVERLAP')
    new_cols = ['rule', 'rule_definition', 'TOTAL_% OVERLAP'] + [col for col in cols if col not in ['rule', 'rule_definition']]
    report_df = report_df[new_cols]
    
    print("Monthly overlap report created")
    return report_df

def save_report_to_excel(report_df, output_path="reports/monthly_rule_overlap_report.xlsx"):
    """Save the report to an Excel file with proper formatting"""
    
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
    
    # Create workbook and get active sheet
    wb = Workbook()
    ws = wb.active
    ws.title = "Monthly Rule Overlap"
    
    # Write headers
    headers = list(report_df.columns)
    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.value = header
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="C9DAF8", end_color="C9DAF8", fill_type="solid")
        cell.alignment = Alignment(horizontal='center')
    
    # Write data
    for row_idx, row in enumerate(dataframe_to_rows(report_df, index=False, header=False), 2):
        for col_idx, value in enumerate(row, 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            
            # Format cells based on content
            if headers[col_idx-1].startswith('%') or 'OVERLAP' in headers[col_idx-1]:
                if isinstance(value, (int, float)):
                    # Format as percentage
                    cell.value = value / 100  # Convert to decimal
                    cell.number_format = '0.0%'
                else:
                    cell.value = value
            else:
                cell.value = value
    
    # Adjust column widths
    for column in ws.columns:
        max_length = 0
        column_letter = column[0].column_letter
        
        for cell in column:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        
        adjusted_width = max(max_length + 2, 10)
        ws.column_dimensions[column_letter].width = adjusted_width
    
    # Save workbook
    wb.save(output_path)
    print(f"Report saved to {output_path}")
    
    return output_path

def visualize_rule_overlaps(monthly_stats, rule_overlaps, all_rules, last_month):
    """Create visualizations for rule overlaps"""
    
    viz_dir = "visualizations/rule_overlaps"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Filter active rules (those with at least one KYC in the last month)
    active_rules = [rule for rule in all_rules 
                   if monthly_stats[last_month][rule]['kyc_count'] > 0]
    
    # Create overlap heatmap
    plt.figure(figsize=(12, 10))
    
    # Build overlap matrix
    overlap_matrix = np.zeros((len(active_rules), len(active_rules)))
    
    for i, rule1 in enumerate(active_rules):
        for j, rule2 in enumerate(active_rules):
            if rule1 != rule2:
                if rule2 in rule_overlaps[last_month][rule1]:
                    overlap_pct = rule_overlaps[last_month][rule1][rule2]['overlap_pct']
                    overlap_matrix[i, j] = overlap_pct
    
    # Plot heatmap
    sns.heatmap(overlap_matrix, 
                xticklabels=active_rules, 
                yticklabels=active_rules,
                cmap="YlGnBu", 
                vmin=0, 
                vmax=100,
                annot=True, 
                fmt=".1f")
    
    plt.title(f'Rule Overlap Percentages - {last_month}')
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/rule_overlap_heatmap_{last_month}.png', dpi=300)
    plt.close()
    
    print(f"Overlap heatmap saved to {viz_dir}/rule_overlap_heatmap_{last_month}.png")
    
    # Create horizontal bar chart for a specific rule (e.g., TRP_0001)
    if "TRP_0001" in active_rules:
        create_specific_rule_overlap_chart("TRP_0001", monthly_stats, rule_overlaps, active_rules, last_month, viz_dir)
    
    return viz_dir

def create_specific_rule_overlap_chart(target_rule, monthly_stats, rule_overlaps, active_rules, month, viz_dir):
    """Create a horizontal bar chart showing overlaps for a specific rule"""
    
    # Get overlap data
    overlaps = []
    for other_rule in active_rules:
        if other_rule != target_rule and other_rule in rule_overlaps[month][target_rule]:
            overlap_pct = rule_overlaps[month][target_rule][other_rule]['overlap_pct']
            overlaps.append((other_rule, overlap_pct))
    
    # Sort by overlap percentage
    overlaps.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 10
    top_overlaps = overlaps[:10]
    
    # Skip if no overlaps
    if not top_overlaps:
        print(f"No overlaps found for {target_rule}")
        return
    
    # Create horizontal bar chart
    plt.figure(figsize=(10, 6))
    
    rules = [x[0] for x in top_overlaps]
    pcts = [x[1] for x in top_overlaps]
    
    # Reverse lists for better visualization (highest at top)
    rules.reverse()
    pcts.reverse()
    
    plt.barh(rules, pcts, color='skyblue')
    plt.xlabel('Overlap Percentage (%)')
    plt.ylabel('Rule')
    plt.title(f'Top Rules Overlapping with {target_rule} - {month}')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f'{viz_dir}/{target_rule}_top_overlaps_{month}.png', dpi=300)
    plt.close()
    
    print(f"Specific rule overlap chart saved to {viz_dir}/{target_rule}_top_overlaps_{month}.png")

def analyze_single_rule(transaction_data, monthly_stats, rule_overlaps, target_rule, all_months):
    """Analyze a single rule in detail"""
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS FOR RULE: {target_rule}")
    print(f"{'='*80}")
    
    # Check if rule exists
    if target_rule not in transaction_data['alert_rules'].unique():
        print(f"Rule {target_rule} not found in transaction data")
        return
    
    # Get rule description
    rule_desc = ""
    rule_rows = transaction_data[transaction_data['alert_rules'] == target_rule]
    if not rule_rows.empty and 'Rule description' in rule_rows.columns:
        rule_desc = rule_rows['Rule description'].iloc[0]
    
    print(f"Rule Description: {rule_desc}")
    print()
    
    # Print monthly statistics
    print("Monthly Statistics:")
    print("-" * 60)
    print(f"{'Month':<10} {'KYC Count':<10} {'Alert Count':<12} {'TP Rate':<10} {'FP Rate':<10}")
    print("-" * 60)
    
    for month in all_months:
        if target_rule in monthly_stats[month]:
            stats = monthly_stats[month][target_rule]
            kyc_count = stats['kyc_count']
            alert_count = stats['alert_count']
            tp_pct = stats['tp_pct']
            fp_pct = stats['fp_pct']
            
            print(f"{month:<10} {kyc_count:<10} {alert_count:<12} {tp_pct:<10.1f}% {fp_pct:<10.1f}%")
    
    print()
    
    # Print overlap statistics for most recent month
    last_month = all_months[-1]
    if target_rule in rule_overlaps[last_month]:
        print(f"Overlap Analysis for {last_month}:")
        print("-" * 70)
        
        overlaps = []
        for other_rule, data in rule_overlaps[last_month][target_rule].items():
            if other_rule != 'TOTAL_OVERLAP':
                overlaps.append((other_rule, data['overlap_count'], data['overlap_pct']))
        
        # Sort by overlap percentage
        overlaps.sort(key=lambda x: x[2], reverse=True)
        
        # Print total overlap
        total_overlap = rule_overlaps[last_month][target_rule].get('TOTAL_OVERLAP', {})
        total_count = total_overlap.get('overlap_count', 0)
        total_pct = total_overlap.get('overlap_pct', 0)
        
        print(f"TOTAL OVERLAP: {total_count} KYCs ({total_pct:.1f}%)")
        print()
        
        print(f"{'Rule':<12} {'Overlap Count':<15} {'Overlap %':<10}")
        print("-" * 70)
        
        for other_rule, count, pct in overlaps[:10]:  # Top 10
            print(f"{other_rule:<12} {count:<15} {pct:<10.1f}%")
    
    print()
    
    # Print status distribution
    rule_data = transaction_data[transaction_data['alert_rules'] == target_rule]
    status_counts = rule_data['status'].value_counts()
    
    print("Status Distribution:")
    print("-" * 30)
    for status, count in status_counts.items():
        print(f"{status:<15} {count:<10}")
    
    # Create visualization directory
    viz_dir = f"visualizations/rule_analysis/{target_rule}"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create monthly trend chart
    if len(all_months) > 1:
        plt.figure(figsize=(10, 6))
        
        # KYC counts by month
        kyc_counts = [monthly_stats[month][target_rule]['kyc_count'] for month in all_months]
        
        plt.plot(all_months, kyc_counts, 'o-', linewidth=2, markersize=8, label='KYC Count')
        
        plt.title(f'Monthly KYC Alert Trend for {target_rule}')
        plt.xlabel('Month')
        plt.ylabel('Number of KYCs')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/monthly_trend.png', dpi=300)
        plt.close()
        
        print(f"Monthly trend chart saved to {viz_dir}/monthly_trend.png")
    
    # Create overlap chart for most recent month
    if target_rule in rule_overlaps[last_month]:
        overlaps = []
        for other_rule, data in rule_overlaps[last_month][target_rule].items():
            if other_rule != 'TOTAL_OVERLAP':
                overlaps.append((other_rule, data['overlap_pct']))
        
        # Sort by overlap percentage
        overlaps.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 10
        top_overlaps = overlaps[:10]
        
        if top_overlaps:
            plt.figure(figsize=(10, 6))
            
            rules = [x[0] for x in top_overlaps]
            pcts = [x[1] for x in top_overlaps]
            
            # Reverse lists for better visualization (highest at top)
            rules.reverse()
            pcts.reverse()
            
            plt.barh(rules, pcts, color='skyblue')
            plt.xlabel('Overlap Percentage (%)')
            plt.ylabel('Rule')
            plt.title(f'Top Rules Overlapping with {target_rule} - {last_month}')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(f'{viz_dir}/top_overlaps.png', dpi=300)
            plt.close()
            
            print(f"Overlap chart saved to {viz_dir}/top_overlaps.png")
    
    return viz_dir

def main():
    """Main function to execute the rule overlap analysis"""
    
    # Load data with joined rule descriptions
    transaction_data, rule_descriptions = load_transaction_data()
    
    # Prepare data
    transaction_data = prepare_data_for_analysis(transaction_data)
    
    # Calculate monthly statistics (now using the joined rule descriptions)
    monthly_stats, all_rules, all_months, rule_definitions = build_monthly_alert_counts(transaction_data)
    
    # Calculate rule overlaps
    rule_overlaps = calculate_rule_overlap(monthly_stats, all_rules, all_months)
    
    # Create report
    report_df = create_monthly_overlap_report(monthly_stats, rule_overlaps, all_rules, all_months, rule_definitions)
    
    # Save report to Excel
    report_path = save_report_to_excel(report_df)
    
    # Create visualizations
    last_month = all_months[-1]
    viz_dir = visualize_rule_overlaps(monthly_stats, rule_overlaps, all_rules, last_month)
    
    print("\nAnalysis complete!")
    print(f"Report saved to: {report_path}")
    print(f"Visualizations saved to: {viz_dir}")
    
    # Display first few rows of the report for verification
    print("\nReport Preview:")
    preview_cols = ['rule', 'rule_definition', 'TOTAL_% OVERLAP'] + [col for col in report_df.columns if col.startswith('no_of_kyc_alerted_')][:3]
    print(report_df[preview_cols].head())
    
    # Analyze a specific rule
    target_rule = "TRP_0001"  # You can change this to any rule you want to analyze
    analyze_single_rule(transaction_data, monthly_stats, rule_overlaps, target_rule, all_months)

if __name__ == "__main__":
    main()
