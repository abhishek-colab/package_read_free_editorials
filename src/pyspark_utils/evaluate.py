from pyspark.ml.evaluation import RegressionEvaluator
import pyspark


def evaluate_model(
    model: pyspark.ml.pipeline.PipelineModel,
    encoded_df: pyspark.sql.dataframe.DataFrame,
    label_col: str,
    metric: str = "rmse",
) -> float:
    """
    This function evaluates a trained machine learning model

    Args:
        pyspark.ml.pipeline.PipelineModel: trained model pipeline
        pyspark.sql.dataframe.DataFrame: dataset to be evaluated
        str: target column
        metric: metric to be used to evaluate

    Returns:
        float: evaluation result

    """
    predictions = model.transform(encoded_df)
    evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName=metric
    )
    out = evaluator.evaluate(predictions)
    print(f"{metric}  : {out}")

    return out
