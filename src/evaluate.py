import contextlib
import io
import json

import click
import pandas as pd
from pyspark.sql import SparkSession
from pyspark_ai import SparkAI

from src.util import substitute_show_to_json

# Mapping of plot type to fields for comparison
plot_type_fields = {
    'scatter': {'x', 'y', 'orientation', 'xaxis', 'yaxis', 'type'},
    'bar': {'x', 'y', 'orientation', 'xaxis', 'yaxis', 'type'},
    'histogram': {'x', 'y', 'orientation', 'xaxis', 'yaxis', 'type'},
    'box': {'x', 'y', 'orientation', 'xaxis', 'yaxis', 'type'},
    'histogram2d': {'x', 'y', 'xbingroup', 'xaxis', 'ybingroup', 'yaxis', 'type'},
    'histogram2dcontour': {'x', 'y', 'xaxis', 'yaxis', 'type'},
    'scattergl': {'x', 'y', 'xaxis', 'yaxis', 'type'},
    'scattergeo': {'type', 'lat', 'lon'},
    'pie': {'labels', 'type', 'values', 'domain'},
    'densitymapbox': {'z', 'lat', 'lon', 'subplot', 'type'}
}


def is_same_mapping(golden_keys, golden_values, predicted_keys, predicted_values):
    """
    Compares two key-value mappings. If the keys in the golden mapping are strings,
    they are matched in a case-insensitive manner to keys in the predicted mapping.
    """

    # Convert the key-value lists to dictionaries
    golden_dict = {key: value for key, value in zip(golden_keys, golden_values)}
    predicted_dict = {key: value for key, value in zip(predicted_keys, predicted_values)}

    # For each key in the golden dictionary, we need to ensure:
    # 1. There's a corresponding key in the predicted dictionary
    # 2. The values match for these keys.
    for g_key, g_value in golden_dict.items():
        matched = False

        for p_key, p_value in predicted_dict.items():
            # Check if the key is a string and then do a case-insensitive comparison.
            # Otherwise, do a direct comparison.
            if isinstance(g_key, str) and isinstance(p_key, str):
                keys_match = g_key.lower() == p_key.lower()
            else:
                keys_match = g_key == p_key

            if keys_match and g_value == p_value:
                matched = True
                break

        if not matched:
            return False

    return True


def is_same_mapping_3(golden_keys1, golden_keys2, golden_values, predicted_keys1, predicted_keys2,
                      predicted_values):
    """
    Compare two sets of triple key-value mappings for exact matches, disregarding the order of keys.
    """
    golden_dict = {(k1, k2): v for k1, k2, v in zip(golden_keys1, golden_keys2, golden_values)}
    predicted_dict = {(k1, k2): v for k1, k2, v in
                      zip(predicted_keys1, predicted_keys2, predicted_values)}

    # Check if both dictionaries have the same keys and values, disregarding order.
    return set(golden_dict.items()) == set(predicted_dict.items())


def evaluate(golden, predicted):
    """
    Compare the golden and predicted plot metadata.

    This function performs a series of checks to determine if the predicted plot metadata
    matches the golden standard. Specifically, it checks:
    1. If the plot types match.
    2. For the 'densitymapbox' plot type, it checks if 'lat', 'lon', and 'z' fields match.
    3. For other plot types, it verifies 'x', 'y', 'lat', 'lon', 'labels', and 'values' fields if any.
    4. It also checks other fields specified in plot_type_fields for exact match.
    """
    if predicted['type'] != golden['type']:
        return False

    golden_type = golden['type']

    # Define a helper function to simplify the field check process.
    def check_fields(field1, field2):
        return set([field1, field2]).issubset(plot_type_fields[golden_type]) and \
               not is_same_mapping(golden[field1], golden[field2], predicted[field1],
                                   predicted[field2])

    if golden_type == 'densitymapbox':
        if not is_same_mapping_3(golden['lat'], golden['lon'], golden['z'],
                                 predicted['lat'], predicted['lon'], predicted['z']):
            return False
    else:
        if any([check_fields('x', 'y'), check_fields('lat', 'lon'),
                check_fields('labels', 'values')]):
            return False

    # Check other fields
    for field in plot_type_fields[golden_type]:
        if field not in ['x', 'y', 'z', 'lat', 'lon', 'labels', 'values']:
            if field not in predicted or predicted[field] != golden[field]:
                return False

    return True


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
        test_cases = [test_case for test_case in all_test_cases if
                      test_case['dataset'] == dataset_url]
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

            if not evaluate(golden_plots[uuid]['data'], predicted['data'][0]):
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
