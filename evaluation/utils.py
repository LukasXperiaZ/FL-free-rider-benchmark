def get_output_path(dataset, n_clients, perc_malicious, attack, detection):
    sep = "--"
    return f"{dataset.value}{sep}n={n_clients}{sep}{perc_malicious}%{sep}{attack.value}{sep}{detection.value}"