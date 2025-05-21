# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Set visualization style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Create output directory for visualizations
os.makedirs('rule_threshold_visualizations', exist_ok=True)
===========================================================
# Load transaction data
def load_data():
    """Load all data from Excel file"""
    # Read transaction data
    transaction_data = pd.read_excel('transaction_dummy_data_10k_final.xlsx', 
                                     sheet_name='transaction_dummy_data_10k')
    
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
    
    return transaction_data, rule_descriptions

# Load the data
transaction_data, rule_descriptions = load_data()

# Explore the data
print(f"Transaction data shape: {transaction_data.shape}")
print(f"Rule descriptions shape: {rule_descriptions.shape}")

# Check the columns
print("\nTransaction data columns:")
print(transaction_data.columns.tolist())

# Check rule patterns and frequencies
print("\nRule patterns:")
print(transaction_data['rule_pattern'].unique())
print("\nRule frequencies:")
print(transaction_data['rule_frequency'].unique())

# Check alert statuses
print("\nAlert statuses:")
print(transaction_data['status'].value_counts())

# Check alert entity distribution
print("\nTriggered on distribution:")
print(transaction_data['triggered_on'].value_counts())

# Look at rule descriptions
print("\nSample rule descriptions:")
print(rule_descriptions.head())
===========================================================
# Map thresholds to rules
def map_thresholds_to_rules():
    """Map thresholds to rules from rule descriptions"""
    # Create a dictionary mapping rule number to threshold
    threshold_map = {}
    
    for _, row in rule_descriptions.iterrows():
        rule_no = row['Rule no.']
        threshold = row['Current threshold']
        # Convert threshold to numeric if possible
        try:
            # Check if threshold contains symbols like '>' or '>='
            if isinstance(threshold, str) and ('>' in threshold):
                # Extract the numeric part
                threshold = float(threshold.replace('>', '').replace('=', '').strip())
            else:
                threshold = float(threshold)
        except (ValueError, TypeError):
            # If conversion fails, keep as is
            pass
        
        threshold_map[rule_no] = threshold
    
    return threshold_map

# Get threshold mapping
rule_thresholds = map_thresholds_to_rules()

# Display a few thresholds
print("Sample rule thresholds:")
for rule, threshold in list(rule_thresholds.items())[:5]:
    print(f"Rule {rule}: {threshold}")
==============================================================
def create_rule_scatter_plots(transaction_data, rule_descriptions, rule_thresholds):
    """Create scatter plots for each rule, showing data points relative to thresholds"""
    
    # Get all unique rules
    unique_rules = transaction_data['alert_rules'].unique()
    
    # For each rule, create a scatter plot
    for rule in unique_rules:
        try:
            print(f"\nProcessing rule: {rule}")
            
            # Filter data for this rule
            rule_data = transaction_data[transaction_data['alert_rules'] == rule].copy()
            
            # Skip if no closed alerts (no TP/FP classification)
            closed_rule_data = rule_data[rule_data['status'].isin(['Closed TP', 'Closed FP'])]
            if len(closed_rule_data) == 0:
                print(f"  No closed alerts for rule {rule}. Skipping.")
                continue
            
            # Get rule information
            rule_pattern = rule_data['rule_pattern'].iloc[0] if not rule_data['rule_pattern'].empty else "Unknown"
            rule_frequency = rule_data['rule_frequency'].iloc[0] if not rule_data['rule_frequency'].empty else "Unknown"
            rule_threshold = rule_thresholds.get(rule, "Unknown")
            
            # Get rule description
            rule_desc = ""
            rule_desc_row = rule_descriptions[rule_descriptions['Rule no.'] == rule]
            if not rule_desc_row.empty:
                rule_desc = rule_desc_row['Rule description'].iloc[0]
            
            print(f"  Pattern: {rule_pattern}, Frequency: {rule_frequency}")
            print(f"  Threshold: {rule_threshold}")
            print(f"  Description: {rule_desc}")
            
            # Determine the appropriate metric based on rule pattern
            if isinstance(rule_pattern, str):
                if "Volume" in rule_pattern:
                    create_volume_rule_plot(rule, rule_data, rule_pattern, rule_frequency, rule_threshold, rule_desc)
                elif "Velocity" in rule_pattern:
                    create_velocity_rule_plot(rule, rule_data, rule_pattern, rule_frequency, rule_threshold, rule_desc)
                elif "1 to Many" in rule_pattern:
                    create_one_to_many_rule_plot(rule, rule_data, rule_pattern, rule_frequency, rule_threshold, rule_desc)
                elif "Many to 1" in rule_pattern:
                    create_many_to_one_rule_plot(rule, rule_data, rule_pattern, rule_frequency, rule_threshold, rule_desc)
                else:
                    print(f"  Unrecognized rule pattern: {rule_pattern}. Using default visualization.")
                    create_default_rule_plot(rule, rule_data, rule_pattern, rule_frequency, rule_threshold, rule_desc)
            else:
                print(f"  Rule pattern not available. Using default visualization.")
                create_default_rule_plot(rule, rule_data, rule_pattern, rule_frequency, rule_threshold, rule_desc)
                
        except Exception as e:
            print(f"  Error processing rule {rule}: {e}")
            import traceback
            traceback.print_exc()
=========================================================================================================================
def create_volume_rule_plot(rule, rule_data, rule_pattern, rule_frequency, rule_threshold, rule_desc):
    """Create scatter plot for Volume pattern rules"""
    print("  Creating Volume rule plot...")
    
    # Determine alert entity ID based on triggered_on
    rule_data['alert_entity_id'] = rule_data.apply(
        lambda x: x['sender_kyc_id_no'] if x['triggered_on'] == 'sender' else x['receiver_kyc_id_no'], 
        axis=1
    )
    
    # Determine the appropriate volume field
    # Try different possible column names for transaction volume
    volume_columns = [col for col in rule_data.columns if 'volume' in col.lower() or 'amount' in col.lower() or 'value' in col.lower()]
    
    if 'usd_value' in rule_data.columns:
        volume_field = 'usd_value'
    elif volume_columns:
        volume_field = volume_columns[0]
    else:
        print("  No suitable volume field found. Using a dummy value.")
        rule_data['dummy_volume'] = 1.0
        volume_field = 'dummy_volume'
    
    # Group by alert entity and calculate aggregated metrics
    if rule_frequency == 'daily':
        # Add date column
        rule_data['transaction_date'] = rule_data['transaction_date_time_local'].dt.date
        
        # Group by entity and date
        agg_data = rule_data.groupby(['alert_entity_id', 'transaction_date']).agg({
            volume_field: 'sum',
            'alert_entity_id': 'count',
            'status': lambda x: x.mode()[0] if not x.empty else 'Unknown'
        }).reset_index()
        
        agg_data.columns = ['alert_entity_id', 'transaction_date', 'total_volume', 'transaction_count', 'status']
        x_label = 'Transaction Date'
        x_field = 'transaction_date'
        
    elif rule_frequency == 'Weekly':
        # Add week column
        rule_data['transaction_week'] = rule_data['transaction_date_time_local'].dt.isocalendar().week
        rule_data['transaction_year'] = rule_data['transaction_date_time_local'].dt.isocalendar().year
        
        # Group by entity and week
        agg_data = rule_data.groupby(['alert_entity_id', 'transaction_year', 'transaction_week']).agg({
            volume_field: 'sum',
            'alert_entity_id': 'count',
            'status': lambda x: x.mode()[0] if not x.empty else 'Unknown'
        }).reset_index()
        
        agg_data.columns = ['alert_entity_id', 'transaction_year', 'transaction_week', 
                            'total_volume', 'transaction_count', 'status']
        agg_data['year_week'] = agg_data['transaction_year'].astype(str) + '-' + agg_data['transaction_week'].astype(str)
        x_label = 'Year-Week'
        x_field = 'year_week'
        
    elif rule_frequency == 'Monthly':
        # Add month column
        rule_data['transaction_month'] = rule_data['transaction_date_time_local'].dt.month
        rule_data['transaction_year'] = rule_data['transaction_date_time_local'].dt.year
        
        # Group by entity and month
        agg_data = rule_data.groupby(['alert_entity_id', 'transaction_year', 'transaction_month']).agg({
            volume_field: 'sum',
            'alert_entity_id': 'count',
            'status': lambda x: x.mode()[0] if not x.empty else 'Unknown'
        }).reset_index()
        
        agg_data.columns = ['alert_entity_id', 'transaction_year', 'transaction_month', 
                            'total_volume', 'transaction_count', 'status']
        agg_data['year_month'] = agg_data['transaction_year'].astype(str) + '-' + agg_data['transaction_month'].astype(str)
        x_label = 'Year-Month'
        x_field = 'year_month'
        
    else:
        # If frequency is unknown, just group by entity
        agg_data = rule_data.groupby('alert_entity_id').agg({
            volume_field: 'sum',
            'alert_entity_id': 'count',
            'status': lambda x: x.mode()[0] if not x.empty else 'Unknown'
        }).reset_index()
        
        agg_data.columns = ['alert_entity_id', 'total_volume', 'transaction_count', 'status']
        
        # Create a dummy ordering field
        agg_data['entity_order'] = range(len(agg_data))
        x_label = 'Entity Order'
        x_field = 'entity_order'
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    scatter = sns.scatterplot(
        data=agg_data,
        x=x_field,
        y='total_volume',
        hue='status',
        size='transaction_count',
        sizes=(50, 300),
        alpha=0.7,
        palette={'Closed TP': 'green', 'Closed FP': 'red'}
    )
    
    # Add a threshold line if available and numeric
    if isinstance(rule_threshold, (int, float)):
        plt.axhline(y=rule_threshold, color='blue', linestyle='--', label=f'Threshold: {rule_threshold}')
    
    # Set title and labels
    plt.title(f'Rule {rule}: {rule_desc}\n(Pattern: {rule_pattern}, Frequency: {rule_frequency})')
    plt.xlabel(x_label)
    plt.ylabel('Total Volume')
    
    # Add legend and adjust plot elements
    plt.legend(title='Status')
    plt.grid(True, alpha=0.3)
    
    # Handle x-axis for categorical variables
    if x_field in ['year_week', 'year_month', 'entity_order']:
        if len(agg_data[x_field].unique()) > 20:
            # If too many categories, only show a subset
            plt.xticks(rotation=45, ha='right')
            every_nth = max(1, len(agg_data[x_field].unique()) // 20)
            for n, label in enumerate(plt.gca().xaxis.get_ticklabels()):
                if n % every_nth != 0:
                    label.set_visible(False)
        else:
            plt.xticks(rotation=45, ha='right')
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(f'rule_threshold_visualizations/rule_{rule}_volume.png')
    plt.close()
    
    print(f"  Created Volume rule plot for {rule}")
=================================================================================
def create_velocity_rule_plot(rule, rule_data, rule_pattern, rule_frequency, rule_threshold, rule_desc):
    """Create scatter plot for Velocity pattern rules"""
    print("  Creating Velocity rule plot...")
    
    # Determine alert entity ID based on triggered_on
    rule_data['alert_entity_id'] = rule_data.apply(
        lambda x: x['sender_kyc_id_no'] if x['triggered_on'] == 'sender' else x['receiver_kyc_id_no'], 
        axis=1
    )
    
    # Determine time period based on rule frequency
    if rule_frequency == 'daily':
        # Add date column
        rule_data['time_period'] = rule_data['transaction_date_time_local'].dt.date
        time_period_label = 'Date'
        
    elif rule_frequency == 'Weekly':
        # Create a year-week identifier
        rule_data['year'] = rule_data['transaction_date_time_local'].dt.isocalendar().year
        rule_data['week'] = rule_data['transaction_date_time_local'].dt.isocalendar().week
        rule_data['time_period'] = rule_data['year'].astype(str) + '-W' + rule_data['week'].astype(str)
        time_period_label = 'Year-Week'
        
    elif rule_frequency == 'Monthly':
        # Create a year-month identifier
        rule_data['year'] = rule_data['transaction_date_time_local'].dt.year
        rule_data['month'] = rule_data['transaction_date_time_local'].dt.month
        rule_data['time_period'] = rule_data['year'].astype(str) + '-' + rule_data['month'].astype(str)
        time_period_label = 'Year-Month'
        
    else:
        # Default to date if frequency is unknown
        rule_data['time_period'] = rule_data['transaction_date_time_local'].dt.date
        time_period_label = 'Date'
    
    # Count transactions per entity per time period
    velocity_data = rule_data.groupby(['alert_entity_id', 'time_period']).agg({
        'transaction_id': 'count',  # Assuming there's a transaction_id column
        'status': lambda x: x.mode()[0] if not x.empty else 'Unknown',
        'alert_entity_id': 'size'  # Get count of transactions
    }).reset_index()
    
    velocity_data.columns = ['alert_entity_id', 'time_period', 'transaction_count', 'status', 'total_transactions']
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    scatter = sns.scatterplot(
        data=velocity_data,
        x='time_period',
        y='transaction_count',
        hue='status',
        size='total_transactions',
        sizes=(50, 300),
        alpha=0.7,
        palette={'Closed TP': 'green', 'Closed FP': 'red'}
    )
    
    # Add a threshold line if available and numeric
    if isinstance(rule_threshold, (int, float)):
        plt.axhline(y=rule_threshold, color='blue', linestyle='--', label=f'Threshold: {rule_threshold}')
    
    # Set title and labels
    plt.title(f'Rule {rule}: {rule_desc}\n(Pattern: {rule_pattern}, Frequency: {rule_frequency})')
    plt.xlabel(time_period_label)
    plt.ylabel('Transaction Count (Velocity)')
    
    # Add legend and adjust plot elements
    plt.legend(title='Status')
    plt.grid(True, alpha=0.3)
    
    # Handle x-axis for numerous time periods
    if len(velocity_data['time_period'].unique()) > 20:
        plt.xticks(rotation=45, ha='right')
        # Show only a subset of time periods
        every_nth = max(1, len(velocity_data['time_period'].unique()) // 20)
        for n, label in enumerate(plt.gca().xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
    else:
        plt.xticks(rotation=45, ha='right')
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(f'rule_threshold_visualizations/rule_{rule}_velocity.png')
    plt.close()
    
    print(f"  Created Velocity rule plot for {rule}")
============================================================================
def create_one_to_many_rule_plot(rule, rule_data, rule_pattern, rule_frequency, rule_threshold, rule_desc):
    """Create scatter plot for 1-to-Many pattern rules"""
    print("  Creating 1-to-Many rule plot...")
    
    # Filter for sender triggered alerts
    sender_data = rule_data[rule_data['triggered_on'] == 'sender'].copy()
    
    if sender_data.empty:
        print("  No sender data for this rule. Skipping 1-to-Many plot.")
        return
    
    # Determine time period based on rule frequency
    if rule_frequency == 'daily':
        # Add date column
        sender_data['time_period'] = sender_data['transaction_date_time_local'].dt.date
        time_period_label = 'Date'
        
    elif rule_frequency == 'Weekly':
        # Create a year-week identifier
        sender_data['year'] = sender_data['transaction_date_time_local'].dt.isocalendar().year
        sender_data['week'] = sender_data['transaction_date_time_local'].dt.isocalendar().week
        sender_data['time_period'] = sender_data['year'].astype(str) + '-W' + sender_data['week'].astype(str)
        time_period_label = 'Year-Week'
        
    elif rule_frequency == 'Monthly':
        # Create a year-month identifier
        sender_data['year'] = sender_data['transaction_date_time_local'].dt.year
        sender_data['month'] = sender_data['transaction_date_time_local'].dt.month
        sender_data['time_period'] = sender_data['year'].astype(str) + '-' + sender_data['month'].astype(str)
        time_period_label = 'Year-Month'
        
    else:
        # Default to date if frequency is unknown
        sender_data['time_period'] = sender_data['transaction_date_time_local'].dt.date
        time_period_label = 'Date'
    
    # Count unique receivers per sender per time period
    unique_receivers_data = sender_data.groupby(['sender_kyc_id_no', 'time_period']).agg({
        'receiver_kyc_id_no': pd.Series.nunique,  # Count unique receivers
        'status': lambda x: x.mode()[0] if not x.empty else 'Unknown',
        'transaction_id': 'count'  # Count transactions
    }).reset_index()
    
    unique_receivers_data.columns = ['sender_kyc_id_no', 'time_period', 'unique_receiver_count', 'status', 'transaction_count']
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    scatter = sns.scatterplot(
        data=unique_receivers_data,
        x='time_period',
        y='unique_receiver_count',
        hue='status',
        size='transaction_count',
        sizes=(50, 300),
        alpha=0.7,
        palette={'Closed TP': 'green', 'Closed FP': 'red'}
    )
    
    # Add a threshold line if available and numeric
    if isinstance(rule_threshold, (int, float)):
        plt.axhline(y=rule_threshold, color='blue', linestyle='--', label=f'Threshold: {rule_threshold}')
    
    # Set title and labels
    plt.title(f'Rule {rule}: {rule_desc}\n(Pattern: {rule_pattern}, Frequency: {rule_frequency})')
    plt.xlabel(time_period_label)
    plt.ylabel('Unique Receiver Count')
    
    # Add legend and adjust plot elements
    plt.legend(title='Status')
    plt.grid(True, alpha=0.3)
    
    # Handle x-axis for numerous time periods
    if len(unique_receivers_data['time_period'].unique()) > 20:
        plt.xticks(rotation=45, ha='right')
        # Show only a subset of time periods
        every_nth = max(1, len(unique_receivers_data['time_period'].unique()) // 20)
        for n, label in enumerate(plt.gca().xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
    else:
        plt.xticks(rotation=45, ha='right')
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(f'rule_threshold_visualizations/rule_{rule}_one_to_many.png')
    plt.close()
    
    print(f"  Created 1-to-Many rule plot for {rule}")
=====================================================================================
def create_many_to_one_rule_plot(rule, rule_data, rule_pattern, rule_frequency, rule_threshold, rule_desc):
    """Create scatter plot for Many-to-1 pattern rules"""
    print("  Creating Many-to-1 rule plot...")
    
    # Filter for receiver triggered alerts
    receiver_data = rule_data[rule_data['triggered_on'] == 'receiver'].copy()
    
    if receiver_data.empty:
        print("  No receiver data for this rule. Skipping Many-to-1 plot.")
        return
    
    # Determine time period based on rule frequency
    if rule_frequency == 'daily':
        # Add date column
        receiver_data['time_period'] = receiver_data['transaction_date_time_local'].dt.date
        time_period_label = 'Date'
        
    elif rule_frequency == 'Weekly':
        # Create a year-week identifier
        receiver_data['year'] = receiver_data['transaction_date_time_local'].dt.isocalendar().year
        receiver_data['week'] = receiver_data['transaction_date_time_local'].dt.isocalendar().week
        receiver_data['time_period'] = receiver_data['year'].astype(str) + '-W' + receiver_data['week'].astype(str)
        time_period_label = 'Year-Week'
        
    elif rule_frequency == 'Monthly':
        # Create a year-month identifier
        receiver_data['year'] = receiver_data['transaction_date_time_local'].dt.year
        receiver_data['month'] = receiver_data['transaction_date_time_local'].dt.month
        receiver_data['time_period'] = receiver_data['year'].astype(str) + '-' + receiver_data['month'].astype(str)
        time_period_label = 'Year-Month'
        
    else:
        # Default to date if frequency is unknown
        receiver_data['time_period'] = receiver_data['transaction_date_time_local'].dt.date
        time_period_label = 'Date'
    
    # Count unique senders per receiver per time period
    unique_senders_data = receiver_data.groupby(['receiver_kyc_id_no', 'time_period']).agg({
        'sender_kyc_id_no': pd.Series.nunique,  # Count unique senders
        'status': lambda x: x.mode()[0] if not x.empty else 'Unknown',
        'transaction_id': 'count'  # Count transactions
    }).reset_index()
    
    unique_senders_data.columns = ['receiver_kyc_id_no', 'time_period', 'unique_sender_count', 'status', 'transaction_count']
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    scatter = sns.scatterplot(
        data=unique_senders_data,
        x='time_period',
        y='unique_sender_count',
        hue='status',
        size='transaction_count',
        sizes=(50, 300),
        alpha=0.7,
        palette={'Closed TP': 'green', 'Closed FP': 'red'}
    )
    
    # Add a threshold line if available and numeric
    if isinstance(rule_threshold, (int, float)):
        plt.axhline(y=rule_threshold, color='blue', linestyle='--', label=f'Threshold: {rule_threshold}')
    
    # Set title and labels
    plt.title(f'Rule {rule}: {rule_desc}\n(Pattern: {rule_pattern}, Frequency: {rule_frequency})')
    plt.xlabel(time_period_label)
    plt.ylabel('Unique Sender Count')
    
    # Add legend and adjust plot elements
    plt.legend(title='Status')
    plt.grid(True, alpha=0.3)
    
    # Handle x-axis for numerous time periods
    if len(unique_senders_data['time_period'].unique()) > 20:
        plt.xticks(rotation=45, ha='right')
        # Show only a subset of time periods
        every_nth = max(1, len(unique_senders_data['time_period'].unique()) // 20)
        for n, label in enumerate(plt.gca().xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
    else:
        plt.xticks(rotation=45, ha='right')
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(f'rule_threshold_visualizations/rule_{rule}_many_to_one.png')
    plt.close()
    
    print(f"  Created Many-to-1 rule plot for {rule}")
===================================================================================================
def create_default_rule_plot(rule, rule_data, rule_pattern, rule_frequency, rule_threshold, rule_desc):
    """Create a default scatter plot for rules that don't match specific patterns"""
    print("  Creating default rule plot...")
    
    # Determine alert entity ID based on triggered_on
    rule_data['alert_entity_id'] = rule_data.apply(
        lambda x: x['sender_kyc_id_no'] if x['triggered_on'] == 'sender' else x['receiver_kyc_id_no'], 
        axis=1
    )
    
    # Try to find relevant metrics for this rule
    # Check for commonly used metrics
    possible_metrics = ['usd_value', 'transaction_amount', 'amount', 'value']
    metric_field = None
    
    for metric in possible_metrics:
        if metric in rule_data.columns:
            metric_field = metric
            break
    
    if metric_field is None:
        # If no standard metric found, create a dummy one
        rule_data['transaction_metric'] = 1.0
        metric_field = 'transaction_metric'
    
    # Group by alert entity
    agg_data = rule_data.groupby('alert_entity_id').agg({
        metric_field: ['sum', 'mean'],
        'alert_entity_id': 'count',
        'status': lambda x: x.mode()[0] if not x.empty else 'Unknown'
    }).reset_index()
    
    agg_data.columns = ['alert_entity_id', 'total_metric', 'avg_metric', 'transaction_count', 'status']
    
    # Create a dummy x-axis for ordering
    agg_data['entity_order'] = range(len(agg_data))
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    scatter = sns.scatterplot(
        data=agg_data,
        x='entity_order',
        y='total_metric',
        hue='status',
        size='transaction_count',
        sizes=(50, 300),
        alpha=0.7,
        palette={'Closed TP': 'green', 'Closed FP': 'red'}
    )
    
    # Add a threshold line if available and numeric
    if isinstance(rule_threshold, (int, float)):
        plt.axhline(y=rule_threshold, color='blue', linestyle='--', label=f'Threshold: {rule_threshold}')
    
    # Set title and labels
    plt.title(f'Rule {rule}: {rule_desc}\n(Pattern: {rule_pattern}, Frequency: {rule_frequency})')
    plt.xlabel('Entity Order')
    plt.ylabel(f'Total {metric_field.capitalize()}')
    
    # Add legend and adjust plot elements
    plt.legend(title='Status')
    plt.grid(True, alpha=0.3)
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(f'rule_threshold_visualizations/rule_{rule}_default.png')
    plt.close()
    
    print(f"  Created default rule plot for {rule}")
===========================================================================================
# Execute the visualization process
create_rule_scatter_plots(transaction_data, rule_descriptions, rule_thresholds)

# Print summary
print("\nVisualization process complete!")
print(f"Scatter plots have been created in the 'rule_threshold_visualizations' directory")
