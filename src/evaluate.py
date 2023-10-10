import contextlib
import io
import json
import logging
from io import StringIO

import pandas as pd
from pyspark.sql import SparkSession
from pyspark_ai import SparkAI
from pyspark_ai.ai_utils import AIUtils

from src.util import substitute_show_to_json


def capture_plot_logs(df, desc_plot, spark_ai):
    """Capture plot logs"""
    root_logger = logging.getLogger()
    buffer = StringIO()
    ch = logging.StreamHandler(buffer)
    root_logger.addHandler(ch)

    with spark_ai._logger.disable_code_colorization():
        df.ai.plot(desc_plot)

    log_contents = buffer.getvalue()
    root_logger.removeHandler(ch)
    return log_contents


def is_subset(sub_dict, main_dict):
    """Check if sub_dict is a subset of main_dict"""
    return all(item in main_dict.items() for item in sub_dict.items())


def main():
    spark = SparkSession.builder.getOrCreate()
    spark_ai = SparkAI(spark_session=spark, verbose=True)
    spark_ai.activate()

    with open('data/train/train-golden.json', 'r') as golden_file, \
            open('data/train/train.json', 'r') as test_file:
        golden_data = json.load(golden_file)
        test_cases = json.load(test_file)

    golden_plots = {golden_plot['uuid']: golden_plot['plot_json'] for golden_plot in golden_data}

    err_cnt = 0
    buffer = io.StringIO()
    for test_case in test_cases:
        buffer.seek(0)  # Reset the buffer's position to the beginning.
        buffer.truncate(0)  # Clear the buffer's contents.
        uuid, dataset, plot_desc = test_case['uuid'], test_case['dataset'], test_case['description']
        pdf = pd.read_csv(dataset)
        df = spark_ai._spark.createDataFrame(pdf)
        captured_output = capture_plot_logs(df, plot_desc, spark_ai)
        codeblocks = AIUtils.extract_code_blocks(captured_output)
        sub_codeblocks = substitute_show_to_json(codeblocks)
        code = "\n".join(sub_codeblocks)

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(code, "plot_df-CodeGen-benchmark", "exec"))
        captured_output = buffer.getvalue()[:-1]
        predicted = json.loads(captured_output)

        if not is_subset(golden_plots[uuid]['data'], predicted['data'][0]):
            print(f"[ERROR] {uuid}")
            print("[PREDICTED]")
            print(predicted['data'][0])
            print("[GOLDEN]")
            print(golden_plots[uuid]['data'])
            err_cnt += 1

    buffer.close()
    print(f"{err_cnt} errors detected")
    total_cases = len(test_cases)
    pass_rate = ((total_cases - err_cnt) / total_cases) * 100
    print(f"Pass rate: {pass_rate:.2f}%")


if __name__ == '__main__':
    main()
