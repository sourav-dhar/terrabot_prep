def analyze_specific_rule_overlap(transaction_data, target_rule="TRP_0001"):
    """
    Analyze KYC IDs that alerted on a specific rule and check their overlap with other rules.
    
    Parameters:
    -----------
    transaction_data : DataFrame
        The transaction data containing alert information
    target_rule : str
        The specific rule to analyze (default: "TRP_0001")
    
    Returns:
    --------
    target_kyc_ids : set
        Set of KYC IDs that alerted on the target rule
    rule_overlap_counts : dict
        Dictionary with counts of overlaps with other rules
    """
    print(f"\nAnalyzing KYC IDs that alerted on {target_rule}...")
    
    # Create a dictionary mapping KYC IDs to the rules they triggered
    kyc_alerts = defaultdict(set)
    
    for idx, row in transaction_data.iterrows():
        if row['triggered_on'] == 'sender':
            kyc_id = row['sender_kyc_id_no']
        else:  # receiver
            kyc_id = row['receiver_kyc_id_no']
            
        kyc_alerts[kyc_id].add(row['alert_rules'])
    
    # Find KYC IDs that alerted on the target rule
    target_kyc_ids = {kyc_id for kyc_id, rules in kyc_alerts.items() if target_rule in rules}
    
    print(f"Total KYC IDs that alerted on {target_rule}: {len(target_kyc_ids)}")
    
    # Count how many of these KYC IDs alerted on other rules as well
    kyc_with_other_rules = sum(1 for kyc_id in target_kyc_ids if len(kyc_alerts[kyc_id]) > 1)
    percentage = (kyc_with_other_rules / len(target_kyc_ids)) * 100 if target_kyc_ids else 0
    
    print(f"KYC IDs that alerted on {target_rule} and other rules: {kyc_with_other_rules} ({percentage:.2f}%)")
    print(f"KYC IDs that alerted only on {target_rule}: {len(target_kyc_ids) - kyc_with_other_rules}")
    
    # Count occurrence of other rules for these KYC IDs
    rule_overlap_counts = {}
    for kyc_id in target_kyc_ids:
        for rule in kyc_alerts[kyc_id]:
            if rule != target_rule:
                rule_overlap_counts[rule] = rule_overlap_counts.get(rule, 0) + 1
    
    # Sort by frequency
    sorted_overlaps = sorted(rule_overlap_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop rules that co-occur with {}:".format(target_rule))
    for rule, count in sorted_overlaps[:10]:  # Show top 10
        overlap_percentage = (count / len(target_kyc_ids)) * 100
        print(f"  {rule}: {count} KYC IDs ({overlap_percentage:.2f}% of {target_rule} KYCs)")
    
    # Visualize the distribution of number of rules per KYC ID
    rule_counts = [len(kyc_alerts[kyc_id]) for kyc_id in target_kyc_ids]
    rule_count_distribution = pd.Series(rule_counts).value_counts().sort_index()
    
    print("\nDistribution of number of rules per KYC ID (for KYCs that alerted on {}):".format(target_rule))
    print(rule_count_distribution)
    
    plt.figure(figsize=(10, 6))
    rule_count_distribution.plot(kind='bar')
    plt.title(f'Number of Rules Triggered per KYC ID (for KYCs that alerted on {target_rule})')
    plt.xlabel('Number of Rules')
    plt.ylabel('Count of KYC IDs')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'visualizations/rule_count_dist_{target_rule}.png')
    plt.show()
    
    # Visualize overlap with top co-occurring rules
    if sorted_overlaps:
        top_rules = [rule for rule, _ in sorted_overlaps[:10]]
        counts = [count for _, count in sorted_overlaps[:10]]
        
        plt.figure(figsize=(12, 6))
        plt.barh(top_rules, counts, color='teal', alpha=0.7)
        plt.title(f'Top Rules Co-occurring with {target_rule}')
        plt.xlabel('Number of KYC IDs')
        plt.ylabel('Rule')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'visualizations/rule_cooccurrence_{target_rule}.png')
        plt.show()
    
    return target_kyc_ids, rule_overlap_counts

# Execute the analysis for TRP_0001
trp0001_kyc_ids, trp0001_overlaps = analyze_specific_rule_overlap(transaction_data, "TRP_0001")

# If you want to analyze another rule, just call the function again with a different rule name
# For example:
# trp0002_kyc_ids, trp0002_overlaps = analyze_specific_rule_overlap(transaction_data, "TRP_0002")
