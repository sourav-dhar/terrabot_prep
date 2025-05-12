import pandas as pd
from aml_suppression import generate_suppression_list, export_report_to_csv, generate_html_report, plot_suppression_data

# Load your alerts data
alerts_df = pd.read_csv('your_aml_alerts.csv')

# Generate suppression report with enhanced analysis
report = generate_suppression_list(alerts_df)

# Generate visualizations
plots = plot_suppression_data(report)

# Export comprehensive HTML report
html_report = generate_html_report(report, plots=plots)
with open('aml_suppression_report.html', 'w') as f:
    f.write(html_report)

# Export detailed CSV files
file_paths = export_report_to_csv(report, output_dir='./reports')
