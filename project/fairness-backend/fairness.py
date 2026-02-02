from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference
)
sensitive_feature = X_test["gender"]

metric_frame = MetricFrame(
    metrics={
        "Selection Rate": selection_rate
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_feature
)

print("Selection Rate by Gender:")
print(metric_frame.by_group)

dp_diff = demographic_parity_difference(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_feature
)

print("Demographic Parity Difference:", dp_diff)

eo_diff = equalized_odds_difference(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_feature
)

print("Equal Opportunity Difference:", eo_diff)

sr = metric_frame.by_group["Selection Rate"]
disparate_impact = sr.min() / sr.max()

print("Disparate Impact:", disparate_impact)
