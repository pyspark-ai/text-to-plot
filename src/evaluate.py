import contextlib
import io
import json
import logging
import re

import click
import pandas as pd
from pyspark.sql import SparkSession
from pyspark_ai import SparkAI

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()])

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


def substitute_show_to_json(string):
    return re.sub(r'(\w+)\.show\(\)', r'print(\1.to_json())', string)


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


def evaluate(golden, predicted, is_hard=False):
    """
    Compare the golden and predicted plot metadata.

    This function performs a series of checks to determine if the predicted plot metadata
    matches the golden standard. Specifically, it checks:
    1. If the plot types match.
    2. For the 'densitymapbox' plot type, it checks if 'lat', 'lon', and 'z' fields match.
    3. For other plot types, it verifies 'x', 'y', 'lat', 'lon', 'labels', and 'values' fields if any.
    4. It also checks other fields specified in plot_type_fields for exact match.
    """
    golden_type = golden['type']

    def fields_match(field1, field2):
        direct_match = set([field1, field2]).issubset(plot_type_fields[golden_type]) and \
                       is_same_mapping(golden[field1], golden[field2], predicted[field1],
                                       predicted[field2])

        if direct_match:
            return True

        # Check for swapped values
        if field1 == 'x' and field2 == 'y':
            return is_same_mapping(golden[field1], golden[field2], predicted[field2],
                                   predicted[field1])

        return False

    def check_densitymapbox_fields():
        return is_same_mapping_3(golden['lat'], golden['lon'], golden['z'],
                                 predicted['lat'], predicted['lon'], predicted['z'])

    def check_standard_fields():
        return any(
            [fields_match('x', 'y'), fields_match('lat', 'lon'), fields_match('labels', 'values')])

    def check_additional_fields():
        for field in plot_type_fields[golden_type]:
            if field not in {'x', 'y', 'z', 'lat', 'lon', 'labels', 'values'} and \
                    (field not in predicted or predicted[field] != golden[field]):
                return False
        return True

    if is_hard:
        return check_standard_fields()
    else:
        if predicted['type'] != golden_type:
            return False

        if golden_type == 'densitymapbox':
            if not check_densitymapbox_fields():
                return False
        else:
            if not check_standard_fields():
                return False

        return check_additional_fields()


@click.command()
@click.option("--test-id", type=str, default=None, help="UUID of the test")
@click.option("--dataset-url", type=str, default=None, help="URL of the dataset")
@click.option("--complexity", type=click.Choice(["easy", "hard"]), default=None,
              help="Test mode (easy, hard)")
def main(test_id, dataset_url, complexity):
    spark = SparkSession.builder.getOrCreate()
    spark_ai = SparkAI(spark_session=spark, verbose=True)
    spark_ai.activate()

    with open('data/train/train-golden.json', 'r') as golden_file:
        golden_data = json.load(golden_file)

    with open('data/train/train.json', 'r') as test_file:
        all_test_cases = json.load(test_file)

    test_cases = [
        test_case for test_case in all_test_cases
        if (not test_id or test_case['uuid'] == test_id)  # Filter based on test_id if provided.
           and (not dataset_url or test_case[
            'dataset'] == dataset_url)  # Filter based on dataset_url if provided.
           and (not complexity or test_case.get('complexity') == complexity)
        # Filter based on complexity if provided.
    ]

    golden_plots = {golden_plot['uuid']: golden_plot['plot_json'] for golden_plot in golden_data}

    err_cnt = 0
    buffer = io.StringIO()
    for test_case in test_cases:
        buffer.seek(0)  # Reset the buffer's position to the beginning.
        buffer.truncate(0)  # Clear the buffer's contents.

        uuid, dataset, plot_desc = test_case['uuid'], test_case['dataset'], test_case['description']
        is_hard = test_case['complexity'] == "hard"
        pdf = pd.read_csv(dataset)
        df = spark_ai._spark.createDataFrame(pdf)

        try:
            original_code = df.ai.plot(plot_desc)
            code = substitute_show_to_json(original_code)

            with contextlib.redirect_stdout(buffer):
                exec(compile(code, "plot_df-CodeGen-benchmark", "exec"))
            captured_output = buffer.getvalue()[:-1]
            predicted = json.loads(captured_output)

            if not evaluate(golden_plots[uuid]['data'], predicted['data'][0], is_hard=is_hard):
                logging.error(f"[ERROR] {uuid}")
                logging.info("[PREDICTED]")
                logging.info(predicted['data'][0])
                logging.info("[GOLDEN]")
                logging.info(golden_plots[uuid]['data'])
                err_cnt += 1

        except Exception as e:
            logging.error(f"An error occurred while processing test case {uuid}: {str(e)}")
            err_cnt += 1

    buffer.close()
    logging.info(f"{err_cnt} errors detected")
    total_cases = len(test_cases)
    pass_rate = ((total_cases - err_cnt) / total_cases) * 100
    logging.info(f"Pass rate: {pass_rate:.2f}%")


if __name__ == '__main__':
    main()
