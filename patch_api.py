import os

def replace_in_file(file_path, old_text, new_text):
    with open(file_path, 'r') as f:
        content = f.read()
    content = content.replace(old_text, new_text)
    with open(file_path, 'w') as f:
        f.write(content)

# 1. Update routers/timeseries.py
ts_file = "app/routers/timeseries.py"
replace_in_file(ts_file,
    'host_ip: str = Query(\n        "10.78.33.83",\n        description="Target host IP address",\n    ),\n) -> TimeseriesResponse:',
    'host_ip: str = Query(\n        "10.78.33.83",\n        description="Target host IP address",\n    ),\n    model: str = Query(\n        "STGNN",\n        description="Model to use (STGNN or NEURALPROPHET)",\n    ),\n) -> TimeseriesResponse:')
replace_in_file(ts_file,
    'host_name=host_name,\n        host_ip=host_ip,\n    )',
    'host_name=host_name,\n        host_ip=host_ip,\n        model=model,\n    )')

# 2. Update routers/host_prediction.py
hp_file = "app/routers/host_prediction.py"
replace_in_file(hp_file,
    'host_ip: str = Query(\n        "10.78.33.83",\n        description="Host IP address",\n    ),\n) -> HostPredictionSummaryResponse:',
    'host_ip: str = Query(\n        "10.78.33.83",\n        description="Host IP address",\n    ),\n    model: str = Query(\n        "STGNN",\n        description="Model to use (STGNN or NEURALPROPHET)",\n    ),\n) -> HostPredictionSummaryResponse:')
replace_in_file(hp_file,
    'host_name=host_name,\n        host_ip=host_ip,\n    )',
    'host_name=host_name,\n        host_ip=host_ip,\n        model=model,\n    )')

# 3. Update routers/explainability.py
ex_file = "app/routers/explainability.py"
replace_in_file(ex_file,
    'top_k: int = Query(\n        5,\n        ge=1,\n        le=20,\n        description="Number of top interdependency drivers to return",\n    ),\n) -> ExplainabilityResponse:',
    'top_k: int = Query(\n        5,\n        ge=1,\n        le=20,\n        description="Number of top interdependency drivers to return",\n    ),\n    model: str = Query(\n        "STGNN",\n        description="Model to use (STGNN or NEURALPROPHET)",\n    ),\n) -> ExplainabilityResponse:')
replace_in_file(ex_file,
    'host_ip=host_ip,\n        top_k=top_k,\n    )',
    'host_ip=host_ip,\n        top_k=top_k,\n        model=model,\n    )')

# 4. Update services/timeseries_service.py
ts_svc = "app/services/timeseries_service.py"
replace_in_file(ts_svc,
    'from app.services.data_loader import DataLoader, METRIC_FRIENDLY_NAMES',
    'from app.services.data_loader import DataLoader, METRIC_FRIENDLY_NAMES\nfrom app.services.np_data_loader import NPDataLoader')
replace_in_file(ts_svc,
    'host_ip: str = "10.78.33.83",\n) -> TimeseriesResponse:',
    'host_ip: str = "10.78.33.83",\n    model: str = "STGNN",\n) -> TimeseriesResponse:')
replace_in_file(ts_svc,
    'as_of_date_str = DataLoader.get_as_of_date()',
    'Loader = NPDataLoader if model.upper() == "NEURALPROPHET" else DataLoader\n    as_of_date_str = Loader.get_as_of_date(host_name) if model.upper() == "NEURALPROPHET" else DataLoader.get_as_of_date()')
replace_in_file(ts_svc,
    'node_id = DataLoader.resolve_node_id(metric)',
    'node_id = metric if model.upper() == "NEURALPROPHET" else DataLoader.resolve_node_id(metric)')
replace_in_file(ts_svc,
    'hist_tuples = DataLoader.get_historical_series(node_id, hist_st, hist_et)',
    'hist_tuples = Loader.get_historical_series(node_id, host_name, hist_st, hist_et) if model.upper() == "NEURALPROPHET" else DataLoader.get_historical_series(node_id, hist_st, hist_et)')
replace_in_file(ts_svc,
    'pred_tuples = DataLoader.get_forecast_series(node_id, pred_st, pred_et)',
    'pred_tuples = Loader.get_forecast_series(node_id, host_name, pred_st, pred_et) if model.upper() == "NEURALPROPHET" else DataLoader.get_forecast_series(node_id, pred_st, pred_et)')
replace_in_file(ts_svc,
    'metric=node_id,',
    'metric=node_id if model.upper() != "NEURALPROPHET" else metric,')

# 5. Update services/host_prediction_service.py
hp_svc = "app/services/host_prediction_service.py"
replace_in_file(hp_svc,
    'from app.services.data_loader import DataLoader, METRIC_FRIENDLY_NAMES',
    'from app.services.data_loader import DataLoader, METRIC_FRIENDLY_NAMES\nfrom app.services.np_data_loader import NPDataLoader')
replace_in_file(hp_svc,
    'host_ip: str = "10.78.33.83",\n) -> HostPredictionSummaryResponse:',
    'host_ip: str = "10.78.33.83",\n    model: str = "STGNN",\n) -> HostPredictionSummaryResponse:')
replace_in_file(hp_svc,
    'as_of_str = DataLoader.get_as_of_date()',
    'Loader = NPDataLoader if model.upper() == "NEURALPROPHET" else DataLoader\n        as_of_str = Loader.get_as_of_date(host_name) if model.upper() == "NEURALPROPHET" else DataLoader.get_as_of_date()')
replace_in_file(hp_svc,
    'as_of_date_str = DataLoader.get_as_of_date()',
    'as_of_date_str = Loader.get_as_of_date(host_name) if model.upper() == "NEURALPROPHET" else DataLoader.get_as_of_date()')
replace_in_file(hp_svc,
    'node_id = DataLoader.resolve_node_id(query)',
    'node_id = query if model.upper() == "NEURALPROPHET" else DataLoader.resolve_node_id(query)')
replace_in_file(hp_svc,
    'current_val = DataLoader.get_last_recorded_value(node_id)',
    'current_val = Loader.get_last_recorded_value(node_id, host_name) if model.upper() == "NEURALPROPHET" else DataLoader.get_last_recorded_value(node_id)')
replace_in_file(hp_svc,
    'pred_tuples = DataLoader.get_forecast_series(node_id, pred_st, pred_et)',
    'pred_tuples = Loader.get_forecast_series(node_id, host_name, pred_st, pred_et) if model.upper() == "NEURALPROPHET" else DataLoader.get_forecast_series(node_id, pred_st, pred_et)')
replace_in_file(hp_svc,
    'confidence, risk_level = DataLoader.get_confidence_and_risk(node_id)',
    'confidence, risk_level = Loader.get_confidence_and_risk(node_id, host_name) if model.upper() == "NEURALPROPHET" else DataLoader.get_confidence_and_risk(node_id)')

# 6. Update services/explainability_service.py
ex_svc = "app/services/explainability_service.py"
replace_in_file(ex_svc,
    'from app.services.data_loader import DataLoader, METRIC_FRIENDLY_NAMES',
    'from app.services.data_loader import DataLoader, METRIC_FRIENDLY_NAMES\nfrom app.services.np_data_loader import NPDataLoader')
replace_in_file(ex_svc,
    'top_k: int = 5,\n) -> ExplainabilityResponse:',
    'top_k: int = 5,\n    model: str = "STGNN",\n) -> ExplainabilityResponse:')
replace_in_file(ex_svc,
    'node_id = DataLoader.resolve_node_id(query)',
    'Loader = NPDataLoader if model.upper() == "NEURALPROPHET" else DataLoader\n    node_id = query if model.upper() == "NEURALPROPHET" else DataLoader.resolve_node_id(query)')
replace_in_file(ex_svc,
    'pred_tuples = DataLoader.get_forecast_series(node_id, st_date, et_date)',
    'pred_tuples = Loader.get_forecast_series(node_id, host_name, st_date, et_date) if model.upper() == "NEURALPROPHET" else DataLoader.get_forecast_series(node_id, st_date, et_date)')
replace_in_file(ex_svc,
    'hist_tuples = DataLoader.get_historical_series(node_id, hist_st, today - timedelta(days=1))',
    'hist_tuples = Loader.get_historical_series(node_id, host_name, hist_st, today - timedelta(days=1)) if model.upper() == "NEURALPROPHET" else DataLoader.get_historical_series(node_id, hist_st, today - timedelta(days=1))')
replace_in_file(ex_svc,
    'confidence, risk_level = DataLoader.get_confidence_and_risk(node_id)',
    'confidence, risk_level = Loader.get_confidence_and_risk(node_id, host_name) if model.upper() == "NEURALPROPHET" else DataLoader.get_confidence_and_risk(node_id)')
replace_in_file(ex_svc,
    'raw_drivers = DataLoader.get_top_drivers(node_id, top_k=top_k)',
    'raw_drivers = Loader.get_top_drivers(node_id, host_name, top_k=top_k) if model.upper() == "NEURALPROPHET" else DataLoader.get_top_drivers(node_id, top_k=top_k)')

print("Files patched successfully!")
