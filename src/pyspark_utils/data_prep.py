from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
import pyspark
from typing import List


def prepare_data(
    df_processed: pyspark.sql.dataframe.DataFrame,
    categorical_cols: List[str],
    indexer_final: pyspark.ml.pipeline.PipelineModel = None,
    encoder_final: pyspark.ml.pipeline.PipelineModel = None,
    is_test: bool = True,
) -> (
    pyspark.sql.dataframe.DataFrame,
    pyspark.ml.pipeline.PipelineModel,
    pyspark.ml.pipeline.PipelineModel,
):
    """
    This function prepares data to be fed to a ml model.
    Performs indexing and one hot encoding on categorical cols.

    Args:
        pyspark.sql.dataframe.DataFrame: df
        list[str]: categorical_columns
        pyspark.ml.pipeline.PipelineModel (optional): indexer
        pyspark.ml.pipeline.PipelineModel (optional): encoder

    Returns:
        pyspark.sql.dataframe.DataFrame: encoded dataframe
        pyspark.ml.pipeline.PipelineModel: indexer
        pyspark.ml.pipeline.PipelineModel: encoder

    """
    indexers = [
        StringIndexer(inputCol=col, outputCol=col + "_index").fit(df_processed)
        for col in categorical_cols
    ]
    if is_test:
        # df_processed = df_processed.dropna(subset='duration')
        [indexer.setHandleInvalid("keep") for indexer in indexers]
    indexer_pipeline = Pipeline(stages=indexers)
    if indexer_final is None:
        indexer_final = indexer_pipeline.fit(df_processed)

    indexed_df = indexer_final.transform(df_processed)

    encoder = [
        OneHotEncoder(inputCol=col + "_index", outputCol=col + "_onehot")
        for col in categorical_cols
    ]
    encoder_pipeline = Pipeline(stages=encoder)
    if encoder_final is None:
        encoder_final = encoder_pipeline.fit(indexed_df)
    encoded_df = encoder_final.transform(indexed_df)

    return encoded_df, indexer_final, encoder_final
