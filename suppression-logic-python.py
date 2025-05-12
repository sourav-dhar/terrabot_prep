import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

def generate_suppression_list(df):
    """
    Generate a suppression list based on the given alert data.
    
    If the latest two consecutive closed cases for an entity are both 'Closed FP'
    with different closed dates AND have closure_reason='manual_investigation',
    suppress all open cases and new cases for 60 days.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing alert data with columns:
        - alert_entity_id: the KYC ID on which the alert was generated
        - case_id: the case identifier
        - alert_id: the alert identifier
        - status: the verdict (Closed FP, Closed TP, Open, Paused)
        - created_at: date the alert was created
        - closed_at: date the alert was closed (may be null)
        - closure_reason: reason for closure (may be null)
        
    Returns:
    --------
    dict
        A dictionary containing suppression report data:
        - summary: overall statistics
        - details: detailed lists of suppressed entities, cases, and alerts
        - dataframes: pandas DataFrames for further analysis
        - analysis: additional analytical insights
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Ensure date columns are datetime objects
    for col in ['created_at', 'closed_at']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Track edge cases and validation issues
    validation_issues = []
    
    # Check for missing required columns
    required_columns = ['alert_entity_id', 'case_id', 'alert_id', 'status', 
                        'created_at', 'closed_at', 'closure_reason']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation_issues.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check for NaN values in critical columns
    for col in ['alert_entity_id', 'case_id', 'alert_id', 'status']:
        if col in df.columns and df[col].isna().any():
            validation_issues.append(f"Found {df[col].isna().sum()} NaN values in {col} column")
    
    # Check for future dates
    current_date = pd.Timestamp.now()
    if 'created_at' in df.columns and (df['created_at'] > current_date).any():
        validation_issues.append(f"Found {(df['created_at'] > current_date).sum()} future dates in created_at column")
    if 'closed_at' in df.columns and (df['closed_at'] > current_date).any():
        validation_issues.append(f"Found {(df['closed_at'] > current_date).sum()} future dates in closed_at column")
    
    # Group by case_id and get case-level information
    case_data = []
    
    # Process each case
    for case_id, case_group in df.groupby('case_id'):
        # Get the alert_entity_id (same for all alerts in a case)
        alert_entity_id = case_group['alert_entity_id'].iloc[0]
        
        # Check if all alerts in the case have the same entity_id
        if len(case_group['alert_entity_id'].unique()) > 1:
            validation_issues.append(f"Case {case_id} has multiple entity IDs: {case_group['alert_entity_id'].unique()}")
        
        # Get the case status (using the most prevalent status in the case)
        status = case_group['status'].value_counts().index[0]
        
        # Note: Multiple alerts in a case may have different statuses
        # This is normal behavior, not an issue to track
        
        # Get the most prevalent closure_reason
        closure_reason = None
        if not case_group['closure_reason'].isna().all():
            closure_reason = case_group['closure_reason'].value_counts().index[0]
        
        # Get the earliest created_at date for the case
        created_at = case_group['created_at'].min()
        
        # Get the latest closed_at date for the case (if any alerts are closed)
        closed_at = None
        if not case_group['closed_at'].isna().all():
            closed_at = case_group['closed_at'].max()
        
        # Extract batch information if available
        batch_id = None
        if 'batch_details' in case_group.columns:
            batch_values = case_group['batch_details'].dropna().unique()
            if len(batch_values) > 0:
                batch_id = batch_values[0]
        
        # Add case information to the list
        case_data.append({
            'case_id': case_id,
            'alert_entity_id': alert_entity_id,
            'status': status,
            'created_at': created_at,
            'closed_at': closed_at,
            'closure_reason': closure_reason,
            'batch_id': batch_id,
            'num_alerts': len(case_group)
        })
    
    # Create a dataframe of cases
    cases_df = pd.DataFrame(case_data)
    
    # Initialize dataframes for suppressed cases and suppression periods
    suppressed_cases = pd.DataFrame()
    suppression_periods = pd.DataFrame()
    
    # Initialize tracking for additional analysis
    rule_effectiveness = {}  # Track which rules lead to suppressions
    batch_analysis = {}     # Track which batches have high suppression rates
    
    # Process each entity to find suppression candidates
    for entity_id, entity_cases in cases_df.groupby('alert_entity_id'):
        # Filter out Open cases and sort by closed_at date (most recent first)
        closed_cases = entity_cases[entity_cases['status'] != 'Open'].dropna(subset=['closed_at'])
        closed_cases = closed_cases.sort_values('closed_at', ascending=False).reset_index(drop=True)
        
        # IMPORTANT: Filter for cases with closure_reason='manual_investigation'
        manual_investigation_cases = closed_cases[closed_cases['closure_reason'] == 'manual_investigation']
        
        # Check if we have at least 2 closed cases with manual_investigation
        if len(manual_investigation_cases) >= 2:
            # Check if the latest 2 cases are both 'Closed FP' and have different closed_at dates
            if (manual_investigation_cases.iloc[0]['status'] == 'Closed FP' and 
                manual_investigation_cases.iloc[1]['status'] == 'Closed FP' and 
                manual_investigation_cases.iloc[0]['closed_at'] != manual_investigation_cases.iloc[1]['closed_at']):
                
                # Determine suppression period
                suppression_start = manual_investigation_cases.iloc[0]['closed_at']
                suppression_end = suppression_start + timedelta(days=60)
                
                # Record suppression details for this alert_entity_id
                period = pd.DataFrame({
                    'alert_entity_id': [entity_id],
                    'suppression_start': [suppression_start],
                    'suppression_end': [suppression_end],
                    'trigger_case_1': [manual_investigation_cases.iloc[0]['case_id']],
                    'trigger_case_1_closed_at': [manual_investigation_cases.iloc[0]['closed_at']],
                    'trigger_case_2': [manual_investigation_cases.iloc[1]['case_id']],
                    'trigger_case_2_closed_at': [manual_investigation_cases.iloc[1]['closed_at']]
                })
                suppression_periods = pd.concat([suppression_periods, period], ignore_index=True)
                
                # Find all Open cases to suppress for this entity
                open_cases = entity_cases[entity_cases['status'] == 'Open'].copy()
                if not open_cases.empty:
                    open_cases['suppression_reason'] = 'Existing Open Case'
                    open_cases['suppression_start'] = suppression_start
                    open_cases['suppression_end'] = suppression_end
                    
                # Find all new cases created during the suppression period
                new_cases = entity_cases[(entity_cases['created_at'] >= suppression_start) & 
                                         (entity_cases['created_at'] <= suppression_end)].copy()
                if not new_cases.empty:
                    new_cases['suppression_reason'] = 'New Case in Suppression Period'
                    new_cases['suppression_start'] = suppression_start
                    new_cases['suppression_end'] = suppression_end
                
                # Combine cases to suppress
                entity_suppressed = pd.concat([open_cases, new_cases], ignore_index=True).drop_duplicates(subset=['case_id'])
                suppressed_cases = pd.concat([suppressed_cases, entity_suppressed], ignore_index=True)
                
                # Track batch information for analysis
                if 'batch_id' in entity_suppressed.columns:
                    for batch in entity_suppressed['batch_id'].dropna().unique():
                        if batch in batch_analysis:
                            batch_analysis[batch] += 1
                        else:
                            batch_analysis[batch] = 1
    
    # Get suppressed alerts (all alerts belonging to suppressed cases)
    suppressed_alerts = pd.DataFrame()
    if not suppressed_cases.empty:
        # Filter the original dataframe to only include alerts from suppressed cases
        suppressed_alerts = df[df['case_id'].isin(suppressed_cases['case_id'])].copy()
        
        # Add suppression reason to alerts based on their case
        suppressed_alerts = suppressed_alerts.merge(
            suppressed_cases[['case_id', 'suppression_reason', 'suppression_start', 'suppression_end']],
            on='case_id',
            how='left'
        )
        
        # Track which rules generate the most suppressions
        if 'rule_name' in suppressed_alerts.columns:
            rule_effectiveness = suppressed_alerts['rule_name'].value_counts().to_dict()
    
    # Additional analysis - Suppression efficiency
    total_open_cases = len(cases_df[cases_df['status'] == 'Open'])
    suppressed_open_cases = len(suppressed_cases[suppressed_cases['status'] == 'Open']) if not suppressed_cases.empty else 0
    
    suppression_efficiency = {
        'total_open_cases': total_open_cases,
        'suppressed_open_cases': suppressed_open_cases,
        'suppression_rate': round(suppressed_open_cases / total_open_cases * 100, 2) if total_open_cases > 0 else 0
    }
    
    # Additional analysis - Suppression timeline forecast
    current_date = pd.Timestamp.now()
    upcoming_expirations = []
    
    if not suppression_periods.empty:
        for _, period in suppression_periods.iterrows():
            if period['suppression_end'] > current_date:
                upcoming_expirations.append({
                    'alert_entity_id': period['alert_entity_id'],
                    'suppression_end': period['suppression_end'],
                    'days_until_expiration': (period['suppression_end'] - current_date).days
                })
    
    # Generate summary
    summary = {
        'total_suppressed_entities': len(suppression_periods),
        'total_suppressed_cases': len(suppressed_cases),
        'total_suppressed_alerts': len(suppressed_alerts),
        'validation_issues': len(validation_issues),
        'suppression_efficiency': f"{suppression_efficiency['suppression_rate']}% ({suppressed_open_cases}/{total_open_cases})"
    }
    
    # Organize data for the report
    report = {
        'summary': summary,
        'details': {
            'suppression_periods': suppression_periods.to_dict('records') if not suppression_periods.empty else [],
            'suppressed_cases': suppressed_cases.to_dict('records') if not suppressed_cases.empty else [],
            'suppressed_alerts': suppressed_alerts.to_dict('records') if not suppressed_alerts.empty else [],
            'validation_issues': validation_issues
        },
        'dataframes': {
            'suppression_periods': suppression_periods,
            'suppressed_cases': suppressed_cases,
            'suppressed_alerts': suppressed_alerts,
            'all_cases': cases_df
        },
        'analysis': {
            'suppression_efficiency': suppression_efficiency,
            'rule_effectiveness': rule_effectiveness,
            'batch_analysis': batch_analysis,
            'upcoming_expirations': upcoming_expirations
        }
    }
    
    return report


def export_report_to_csv(report, output_dir='./'):
    """
    Export suppression report to CSV files.
    
    Parameters:
    -----------
    report : dict
        The suppression report generated by generate_suppression_list()
    output_dir : str, optional
        Directory to save the CSV files, defaults to current directory
    
    Returns:
    --------
    dict
        Dictionary with file paths of exported CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Export suppression periods
    periods_file = os.path.join(output_dir, f'suppression_periods_{timestamp}.csv')
    if not report['dataframes']['suppression_periods'].empty:
        report['dataframes']['suppression_periods'].to_csv(periods_file, index=False)
    
    # Export suppressed cases
    cases_file = os.path.join(output_dir, f'suppressed_cases_{timestamp}.csv')
    if not report['dataframes']['suppressed_cases'].empty:
        report['dataframes']['suppressed_cases'].to_csv(cases_file, index=False)
    
    # Export suppressed alerts
    alerts_file = os.path.join(output_dir, f'suppressed_alerts_{timestamp}.csv')
    if not report['dataframes']['suppressed_alerts'].empty:
        report['dataframes']['suppressed_alerts'].to_csv(alerts_file, index=False)
    
    # Export validation issues
    validation_file = os.path.join(output_dir, f'validation_issues_{timestamp}.csv')
    pd.DataFrame(report['details']['validation_issues'], columns=['issue']).to_csv(validation_file, index=False)
    
    # Export summary as a simple CSV
    summary_file = os.path.join(output_dir, f'suppression_summary_{timestamp}.csv')
    pd.DataFrame([report['summary']]).to_csv(summary_file, index=False)
    
    # Export analysis data
    analysis_file = os.path.join(output_dir, f'suppression_analysis_{timestamp}.csv')
    
    # Combine analysis data into a DataFrame
    analysis_data = []
    
    # Suppression efficiency
    for key, value in report['analysis']['suppression_efficiency'].items():
        analysis_data.append({'analysis_type': 'suppression_efficiency', 'metric': key, 'value': value})
    
    # Rule effectiveness
    for rule, count in report['analysis']['rule_effectiveness'].items():
        analysis_data.append({'analysis_type': 'rule_effectiveness', 'metric': rule, 'value': count})
    
    # Batch analysis
    for batch, count in report['analysis']['batch_analysis'].items():
        analysis_data.append({'analysis_type': 'batch_analysis', 'metric': batch, 'value': count})
    
    # Expiration forecast
    for exp in report['analysis']['upcoming_expirations']:
        analysis_data.append({
            'analysis_type': 'expiration_forecast', 
            'metric': exp['alert_entity_id'], 
            'value': exp['days_until_expiration']
        })
    
    pd.DataFrame(analysis_data).to_csv(analysis_file, index=False)
    
    return {
        'summary': summary_file,
        'periods': periods_file,
        'cases': cases_file,
        'alerts': alerts_file,
        'validation': validation_file,
        'analysis': analysis_file
    }


def plot_suppression_data(report):
    """
    Generate visualizations for the suppression report
    
    Parameters:
    -----------
    report : dict
        The suppression report generated by generate_suppression_list()
        
    Returns:
    --------
    dict
        Dictionary with encoded visualization images
    """
    plots = {}
    
    # Set style
    sns.set(style="whitegrid")
    
    # 1. Suppression efficiency pie chart
    if report['analysis']['suppression_efficiency']['total_open_cases'] > 0:
        plt.figure(figsize=(10, 6))
        labels = ['Suppressed', 'Not Suppressed']
        sizes = [
            report['analysis']['suppression_efficiency']['suppressed_open_cases'],
            report['analysis']['suppression_efficiency']['total_open_cases'] - 
            report['analysis']['suppression_efficiency']['suppressed_open_cases']
        ]
        colors = ['#5cb85c', '#d9534f']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Open Cases Suppression Rate')
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots['suppression_rate'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    
    # 2. Rule effectiveness bar chart
    if report['analysis']['rule_effectiveness']:
        plt.figure(figsize=(12, 8))
        rule_data = pd.Series(report['analysis']['rule_effectiveness']).sort_values(ascending=False)
        
        # Take top 10 rules if there are more than 10
        if len(rule_data) > 10:
            rule_data = rule_data.head(10)
            
        sns.barplot(x=rule_data.values, y=rule_data.index)
        plt.title('Top Rules Leading to Suppressions')
        plt.xlabel('Number of Suppressions')
        plt.ylabel('Rule Name')
        plt.tight_layout()
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots['rule_effectiveness'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    
    # 3. Suppression timeline
    if not report['dataframes']['suppression_periods'].empty:
        plt.figure(figsize=(12, 6))
        
        # Create a timeline of suppression periods
        timeline_df = report['dataframes']['suppression_periods'].sort_values('suppression_start')
        
        # Plot each suppression period as a horizontal line
        for idx, row in timeline_df.iterrows():
            plt.plot(
                [row['suppression_start'], row['suppression_end']], 
                [idx, idx], 
                linewidth=8, 
                solid_capstyle='butt'
            )
            plt.text(row['suppression_start'], idx-0.3, row['alert_entity_id'], fontsize=9)
        
        plt.yticks([])
        plt.title('Suppression Timeline')
        plt.xlabel('Date')
        plt.tight_layout()
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots['timeline'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    
    # 4. Upcoming expirations
    if report['analysis']['upcoming_expirations']:
        plt.figure(figsize=(12, 8))
        exp_df = pd.DataFrame(report['analysis']['upcoming_expirations'])
        exp_df = exp_df.sort_values('days_until_expiration')
        
        # Take top 15 if there are more than 15
        if len(exp_df) > 15:
            exp_df = exp_df.head(15)
            
        sns.barplot(x='days_until_expiration', y='alert_entity_id', data=exp_df)
        plt.title('Upcoming Suppression Expirations')
        plt.xlabel('Days Until Expiration')
        plt.ylabel('Entity ID')
        plt.tight_layout()
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots['expirations'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    
    return plots


def generate_html_report(report, title="AML Alert Suppression Report", plots=None):
    """
    Generate an HTML report from the suppression data
    
    Parameters:
    -----------
    report : dict
        The suppression report generated by generate_suppression_list()
    title : str, optional
        Title for the HTML report
    plots : dict, optional
        Dictionary with encoded visualization images
        
    Returns:
    --------
    str
        HTML content of the report
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333366; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #333366; color: white; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .summary-box {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .suppression-period {{ background-color: #e6f7ff; padding: 10px; margin-bottom: 10px; border-left: 4px solid #1890ff; }}
            .dashboard {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
            .chart {{ flex: 1; min-width: 400px; background-color: white; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); padding: 15px; }}
            .issues {{ background-color: #fff0f0; padding: 10px; border-left: 4px solid #ff4d4f; margin-bottom: 20px; }}
            .analysis-section {{ margin-bottom: 30px; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <div class="summary-box">
            <h2>Summary</h2>
            <p>Total Entities with Suppression: {report['summary']['total_suppressed_entities']}</p>
            <p>Total Cases Suppressed: {report['summary']['total_suppressed_cases']}</p>
            <p>Total Alerts Suppressed: {report['summary']['total_suppressed_alerts']}</p>
            <p>Suppression Efficiency: {report['summary']['suppression_efficiency']}</p>
            <p>Validation Issues: {report['summary']['validation_issues']}</p>
        </div>
    """
    
    # Add visualization dashboard if plots are provided
    if plots:
        html += '<h2>Suppression Analysis Dashboard</h2>'
        html += '<div class="dashboard">'
        
        for plot_name, plot_data in plots.items():
            html += f"""
            <div class="chart">
                <h3>{plot_name.replace('_', ' ').title()}</h3>
                <img src="data:image/png;base64,{plot_data}" style="width: 100%;">
            </div>
            """
        
        html += '</div>'
    
    # Add validation issues section
    if report['details']['validation_issues']:
        html += """
        <div class="issues">
            <h2>Validation Issues</h2>
            <ul>
        """
        
        for issue in report['details']['validation_issues']:
            html += f"<li>{issue}</li>"
        
        html += """
            </ul>
        </div>
        """
    
    # Add suppression periods section
    if report['details']['suppression_periods']:
        html += """
        <h2>Suppression Periods</h2>
        <table>
            <tr>
                <th>Entity ID</th>
                <th>Suppression Start</th>
                <th>Suppression End</th>
                <th>Trigger Case 1</th>
                <th>Closed At</th>
                <th>Trigger Case 2</th>
                <th>Closed At</th>
            </tr>
        """
        
        for period in report['details']['suppression_periods']:
            html += f"""
            <tr>
                <td>{period['alert_entity_id']}</td>
                <td>{period['suppression_start']}</td>
                <td>{period['suppression_end']}</td>
                <td>{period['trigger_case_1']}</td>
                <td>{period['trigger_case_1_closed_at']}</td>
                <td>{period['trigger_case_2']}</td>
                <td>{period['trigger_case_2_closed_at']}</td>
            </tr>
            """
        
        html += "</table>"
    
    # Add suppressed cases section
    if report['details']['suppressed_cases']:
        html += """
        <h2>Suppressed Cases</h2>
        <table>
            <tr>
                <th>Case ID</th>
                <th>Entity ID</th>
                <th>Status</th>
                <th>Created At</th>
                <th>Suppression Reason</th>
                <th>Suppression Period</th>
            </tr>
        """
        
        for case in report['details']['suppressed_cases']:
            html += f"""
            <tr>
                <td>{case['case_id']}</td>
                <td>{case['alert_entity_id']}</td>
                <td>{case['status']}</td>
                <td>{case['created_at']}</td>
                <td>{case['suppression_reason']}</td>
                <td>{case.get('suppression_start', '')} to {case.get('suppression_end', '')}</td>
            </tr>
            """
        
        html += "</table>"
    
    # Add suppressed alerts section (limited to first 100 for performance)
    if report['details']['suppressed_alerts']:
        html += """
        <h2>Suppressed Alerts (First 100)</h2>
        <table>
            <tr>
                <th>Alert ID</th>
                <th>Case ID</th>
                <th>Entity ID</th>
                <th>Status</th>
                <th>Created At</th>
                <th>Suppression Reason</th>
            </tr>
        """
        
        for alert in report['details']['suppressed_alerts'][:100]:
            html += f"""
            <tr>
                <td>{alert['alert_id']}</td>
                <td>{alert['case_id']}</td>
                <td>{alert['alert_entity_id']}</td>
                <td>{alert['status']}</td>
                <td>{alert['created_at']}</td>
                <td>{alert.get('suppression_reason', '')}</td>
            </tr>
            """
        
        html += "</table>"
    
    # Add analysis sections
    html += """
    <div class="analysis-section">
        <h2>Advanced Analysis</h2>
    """
    
    # Upcoming expirations
    if report['analysis']['upcoming_expirations']:
        html += """
        <h3>Upcoming Suppression Expirations</h3>
        <table>
            <tr>
                <th>Entity ID</th>
                <th>Suppression End Date</th>
                <th>Days Until Expiration</th>
            </tr>
        """
        
        for exp in sorted(report['analysis']['upcoming_expirations'], key=lambda x: x['days_until_expiration']):
            html += f"""
            <tr>
                <td>{exp['alert_entity_id']}</td>
                <td>{exp['suppression_end']}</td>
                <td>{exp['days_until_expiration']}</td>
            </tr>
            """
        
        html += "</table>"
    
    # Rule effectiveness
    if report['analysis']['rule_effectiveness']:
        html += """
        <h3>Rule Effectiveness Analysis</h3>
        <table>
            <tr>
                <th>Rule Name</th>
                <th>Suppressed Alerts</th>
            </tr>
        """
        
        for rule, count in sorted(report['analysis']['rule_effectiveness'].items(), key=lambda x: x[1], reverse=True):
            html += f"""
            <tr>
                <td>{rule}</td>
                <td>{count}</td>
            </tr>
            """
        
        html += "</table>"
    
    # Batch analysis
    if report['analysis']['batch_analysis']:
        html += """
        <h3>Batch Analysis</h3>
        <table>
            <tr>
                <th>Batch ID</th>
                <th>Suppressed Cases</th>
            </tr>
        """
        
        for batch, count in sorted(report['analysis']['batch_analysis'].items(), key=lambda x: x[1], reverse=True):
            html += f"""
            <tr>
                <td>{batch}</td>
                <td>{count}</td>
            </tr>
            """
        
        html += "</table>"
    
    html += """
    </div>
    </body>
    </html>
    """
    
    return html


# Example usage with sample data
if __name__ == "__main__":
    # Create sample data
    data = [
        {'alert_entity_id': 'KS0341888', 'case_id': 'C001', 'alert_id': 'A001', 'status': 'Closed FP', 'created_at': '2023-01-01', 'closed_at': '2023-01-10', 'closure_reason': 'manual_investigation', 'rule_name': 'many_to_one'},
        {'alert_entity_id': 'KS0341888', 'case_id': 'C001', 'alert_id': 'A002', 'status': 'Closed FP', 'created_at': '2023-01-01', 'closed_at': '2023-01-10', 'closure_reason': 'manual_investigation', 'rule_name': 'high_volume'},
        {'alert_entity_id': 'KS0341888', 'case_id': 'C001', 'alert_id': 'A003', 'status': 'Closed FP', 'created_at': '2023-01-01', 'closed_at': '2023-01-10', 'closure_reason': 'manual_investigation', 'rule_name': 'high_value'},
        
        {'alert_entity_id': 'KS0341888', 'case_id': 'C002', 'alert_id': 'A004', 'status': 'Closed FP', 'created_at': '2023-01-15', 'closed_at': '2023-01-25', 'closure_reason': 'manual_investigation', 'rule_name': 'many_to_one'},
        {'alert_entity_id': 'KS0341888', 'case_id': 'C002', 'alert_id': 'A005', 'status': 'Closed FP', 'created_at': '2023-01-15', 'closed_at': '2023-01-25', 'closure_reason': 'manual_investigation', 'rule_name': 'high_volume'},
        
        {'alert_entity_id': 'KS0341888', 'case_id': 'C003', 'alert_id': 'A006', 'status': 'Open', 'created_at': '2023-01-30', 'closed_at': None, 'closure_reason': None, 'rule_name': 'many_to_one'},
        {'alert_entity_id': 'KS0341888', 'case_id': 'C003', 'alert_id': 'A007', 'status': 'Open', 'created_at': '2023-01-30', 'closed_at': None, 'closure_reason': None, 'rule_name': 'high_volume'},
        
        {'alert_entity_id': 'KS0341888', 'case_id': 'C004', 'alert_id': 'A008', 'status': 'Open', 'created_at': '2023-02-15', 'closed_at': None, 'closure_reason': None, 'rule_name': 'high_value'},
        
        # This case should NOT trigger suppression (Self Transfers instead of manual_investigation)
        {'alert_entity_id': 'KS9876543', 'case_id': 'C005', 'alert_id': 'A009', 'status': 'Closed FP', 'created_at': '2023-01-05', 'closed_at': '2023-01-15', 'closure_reason': 'Self Transfers', 'rule_name': 'one_to_many'},
        {'alert_entity_id': 'KS9876543', 'case_id': 'C006', 'alert_id': 'A010', 'status': 'Closed FP', 'created_at': '2023-01-20', 'closed_at': '2023-01-30', 'closure_reason': 'Self Transfers', 'rule_name': 'one_to_many'},
        {'alert_entity_id': 'KS9876543', 'case_id': 'C007', 'alert_id': 'A011', 'status': 'Open', 'created_at': '2023-02-10', 'closed_at': None, 'closure_reason': None, 'rule_name': 'high_value'},
        
        # Multiple consecutive "Closed FP" cases with manual_investigation
        {'alert_entity_id': 'KS1234567', 'case_id': 'C008', 'alert_id': 'A012', 'status': 'Closed FP', 'created_at': '2023-01-05', 'closed_at': '2023-01-15', 'closure_reason': 'manual_investigation', 'rule_name': 'structuring'},
        {'alert_entity_id': 'KS1234567', 'case_id': 'C008', 'alert_id': 'A013', 'status': 'Closed FP', 'created_at': '2023-01-05', 'closed_at': '2023-01-15', 'closure_reason': 'manual_investigation', 'rule_name': 'structuring'},
        {'alert_entity_id': 'KS1234567', 'case_id': 'C009', 'alert_id': 'A014', 'status': 'Closed FP', 'created_at': '2023-01-20', 'closed_at': '2023-01-25', 'closure_reason': 'manual_investigation', 'rule_name': 'structuring'},
        {'alert_entity_id': 'KS1234567', 'case_id': 'C010', 'alert_id': 'A015', 'status': 'Open', 'created_at': '2023-01-30', 'closed_at': None, 'closure_reason': None, 'rule_name': 'structuring'},
        
        # Edge case: same closed date
        {'alert_entity_id': 'KS7654321', 'case_id': 'C009', 'alert_id': 'A014', 'status': 'Closed FP', 'created_at': '2023-01-05', 'closed_at': '2023-01-15', 'closure_reason': 'manual_investigation', 'rule_name': 'rapid_movement'},
        {'alert_entity_id': 'KS7654321', 'case_id': 'C010', 'alert_id': 'A015', 'status': 'Closed FP', 'created_at': '2023-01-10', 'closed_at': '2023-01-15', 'closure_reason': 'manual_investigation', 'rule_name': 'rapid_movement'},
        {'alert_entity_id': 'KS7654321', 'case_id': 'C011', 'alert_id': 'A016', 'status': 'Open', 'created_at': '2023-01-20', 'closed_at': None, 'closure_reason': None, 'rule_name': 'high_value'},
        
        # Example with batch details
        {'alert_entity_id': 'KS8888888', 'case_id': 'C012', 'alert_id': 'A017', 'status': 'Closed FP', 'created_at': '2023-01-05', 'closed_at': '2023-01-15', 'closure_reason': 'manual_investigation', 'rule_name': 'high_risk_country', 'batch_details': 'batch-1'},
        {'alert_entity_id': 'KS8888888', 'case_id': 'C013', 'alert_id': 'A018', 'status': 'Closed FP', 'created_at': '2023-01-20', 'closed_at': '2023-01-25', 'closure_reason': 'manual_investigation', 'rule_name': 'high_risk_country', 'batch_details': 'batch-2'},
        {'alert_entity_id': 'KS8888888', 'case_id': 'C014', 'alert_id': 'A019', 'status': 'Open', 'created_at': '2023-02-10', 'closed_at': None, 'closure_reason': None, 'rule_name': 'high_value', 'batch_details': 'batch-3'},
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Generate suppression report
    report = generate_suppression_list(df)
    
    # Generate plots
    plots = plot_suppression_data(report)
    
    # Print summary
    print("Suppression Summary:")
    for key, value in report['summary'].items():
        print(f"  {key}: {value}")
    
    # Export to CSV (uncomment to use)
    # file_paths = export_report_to_csv(report)
    # print("\nExported files:")
    # for key, path in file_paths.items():
    #     print(f"  {key}: {path}")
    
    # Generate HTML report (uncomment to use)
    # html_report = generate_html_report(report, plots=plots)
    # with open('suppression_report.html', 'w') as f:
    #     f.write(html_report)
    # print("\nHTML report saved to suppression_report.html")
