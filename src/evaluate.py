import contextlib
import io
import json

import click
import pandas as pd
from pyspark.sql import SparkSession
from pyspark_ai import SparkAI

from src.util import substitute_show_to_json


def is_subset(sub_dict, main_dict):
    """Check if sub_dict is a subset of main_dict"""
    return all(item in main_dict.items() for item in sub_dict.items())


@click.command()
@click.argument("dataset_url", type=str, default=None)
def main(dataset_url):
    spark = SparkSession.builder.getOrCreate()
    spark_ai = SparkAI(spark_session=spark, verbose=True)
    spark_ai.activate()

    with open('data/train/train-golden.json', 'r') as golden_file:
        golden_data = json.load(golden_file)

    with open('data/train/train.json', 'r') as test_file:
        all_test_cases = json.load(test_file)

    if dataset_url is not None:
        # Filter test_cases to only include those with matching dataset field
        test_cases = [test_case for test_case in all_test_cases if test_case['dataset'] == dataset_url]
    else:
        test_cases = all_test_cases

    golden_plots = {golden_plot['uuid']: golden_plot['plot_json'] for golden_plot in golden_data}

    err_cnt = 0
    buffer = io.StringIO()
    for test_case in test_cases:
        buffer.seek(0)  # Reset the buffer's position to the beginning.
        buffer.truncate(0)  # Clear the buffer's contents.

        uuid, dataset, plot_desc = test_case['uuid'], test_case['dataset'], test_case['description']
        pdf = pd.read_csv(dataset)
        df = spark_ai._spark.createDataFrame(pdf)

        try:
            original_code = df.ai.plot(plot_desc)
            code = substitute_show_to_json(original_code)

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

        except Exception as e:
            print(f"An error occurred while processing test case {uuid}: {str(e)}")
            err_cnt += 1

    buffer.close()
    print(f"{err_cnt} errors detected")
    total_cases = len(test_cases)
    pass_rate = ((total_cases - err_cnt) / total_cases) * 100
    print(f"Pass rate: {pass_rate:.2f}%")


if __name__ == '__main__':
    main()
