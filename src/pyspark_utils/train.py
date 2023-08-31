from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
import pyspark
from typing import List


def train_model(
    encoded_df: pyspark.sql.dataframe.DataFrame,
    feature_cols: List[str],
    label_col: str,
    regressor=LinearRegression,
    **kwargs,
) -> pyspark.ml.pipeline.PipelineModel:
    """
    This function trains a machine learning model

    Args:
        pyspark.sql.dataframe.DataFrame: encoded_df
        list[str]: Predictor Columns
        str: target column
        regressor: A regressor from pyspark.ml.regression
        **kwargs: Additional arguments of the regresor

    Returns:
        pyspark.ml.pipeline.PipelineModel: trained model pipeline

    """

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    regressor = regressor(featuresCol="features", labelCol=label_col, **kwargs)
    pipeline = Pipeline(stages=[assembler, regressor])
    model = pipeline.fit(encoded_df)
    return model
