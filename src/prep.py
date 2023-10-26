import contextlib
import hashlib
import io
import json
import re

import pandas as pd

from pyspark.sql import SparkSession
from pyspark_ai import SparkAI

# Constants
INCLUDE_KEYS = [
    "x", "y", "xaxis", "yaxis", "type", "orientation",
    "lat", "lon", "z",  # density_mapbox
    "domain", "labels", "values",  # pie
    "xbingroup", "ybingroup",  # bin
]

TRAIN_DATASETS = [
    "https://raw.githubusercontent.com/plotly/datasets/master/1962_2006_walmart_store_openings.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/2011_february_aa_flight_paths.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/US-shooting-incidents.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/titanic.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv",
]

TEST_DATASETS = [
    "https://raw.githubusercontent.com/plotly/datasets/master/european_turnout.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/mpg.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/medicare.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/volcano_db.csv",
]


def substitute_show_to_json(string):
    return re.sub(r'(\w+)\.show\(\)', r'print(\1.to_json())', string)


def generate_id(dataset, description):
    combined_string = dataset + description
    return hashlib.md5(combined_string.encode()).hexdigest()


def filter_golden_json_data(json_string, include_keys=None):
    data_dict = json.loads(json_string)

    if 'data' not in data_dict:
        raise ValueError("'data' field does not exist in the provided JSON string.")

    if len(data_dict['data']) == 1:
        data_content = data_dict['data'][0]
        if include_keys is None:
            filtered_data = {key: data_content[key] for key in data_content}
        else:
            filtered_data = {key: data_content[key] for key in include_keys if key in data_content}
    else:
        x_values = []
        y_values = []
        plot_type = None
        xaxis = None
        yaxis = None
        orientation = None

        for item in data_dict['data']:
            if 'x' in item:
                x_values.extend(item['x'])
            if 'y' in item:
                y_values.extend(item['y'])
            if plot_type is None and 'type' in item:
                plot_type = item['type']
            if xaxis is None and 'xaxis' in item:
                xaxis = item['xaxis']
            if yaxis is None and 'yaxis' in item:
                yaxis = item['yaxis']
            if orientation is None and 'orientation' in item:
                orientation = item['orientation']

        filtered_data = {
            'x': x_values,
            'y': y_values,
            'type': plot_type,
            'xaxis': xaxis,
            'yaxis': yaxis,
            'orientation': orientation
        }

    return {'data': filtered_data}


def gen_test_case(dataset, mode="train"):
    """
    Generate test cases with uuid -> test.json
    """
    if "1962_2006_walmart_store_openings.csv" in dataset:
        easy_descriptions = [
            "Line plot displaying the number of Walmart stores opened each year from 1962 to 2006.",
            "Bar plot showing the count of Walmart stores by state.",
            "Histogram illustrating the distribution of store openings by month.",
            "Boxplot representing the distribution of store conversions.",
            "Density plot showcasing the distribution of Walmart store openings over the years.",
            "Area plot highlighting the cumulative number of Walmart stores opened from 1962 to 2006.",
            "Scatter plot of Walmart store opened in 1969 using latitude and longitude.",
            "Pie chart showing the proportion of different types of Walmart stores.",
            "Bar plot representing the number of Walmart stores opened each month.",
            "Line plot illustrating the trend of store conversions over the years.",
            "A hexagonal bin plot visualizes the density of Walmart stores opened in August 1984 based on their latitude and longitude.",
            "Scatter plot illustrating Walmart stores in CA that opened in 1990, plotted using latitude and longitude.",
            "Boxplot comparing the distribution of store openings by month, separated by store type.",
            "Density plot illustrating the distribution of store conversions, segmented by state.",
            "Area plot representing the cumulative number of Walmart stores, with each type of store shaded differently."
        ]

        hard_descriptions = [
            "Yearly trend of Walmart store openings from 1996 to 2006.",
            "Top 5 states with the highest number of Walmart store openings.",
            "Proportion of stores by type after 2000.",
            "Proportion of stores by type before 2000.",
            "Distribution of the number of stores opened each year after 2000.",
        ]
    elif "2011_february_aa_flight_paths.csv" in dataset:
        easy_descriptions = [
            'Bar plots showcasing the frequency of AA flights from different starting airports.',
            'Bar plots representing the distribution of ending locations for AA flights.',
            'Histogram illustrating the common starting longitudes for the flights.',
            'Histogram depicting the popular ending latitudes for AA flights.',
            'Boxplot summarizing the range of starting latitudes for all AA flights.',
            'Boxplot visualizing the range of ending longitudes for these flights.',
            'Density plots highlighting the concentration of starting locations.',
            'Density plots emphasizing the concentration of ending locations.',
            'Area plots comparing the frequency of flights based on starting and ending latitudes.',
            'Scatter plots visualizing the correlation between starting and ending locations for the flights.',
            'Pie plots representing the proportion of flights based on their starting airports.'
        ]

        hard_descriptions = [
            "Top 5 airports with the highest number of arrivals.",
            "Top 5 airports with the highest number of departure.",
            "Proportion of top 5 flights based on their starting airports.",
        ]
    elif "2011_us_ag_exports.csv" in dataset:
        easy_descriptions = [
            "Bar plot comparing beef exports across states.",
            "Histogram showing the distribution of poultry exports.",
            "Boxplot representing the spread of dairy exports.",
            "Area plot indicating the trend of wheat exports by state.",
            "Hexagonal bin plot showing the relationship between fruits fresh and fruits processed exports.",
            "Pie chart representing the share of total veggies exported by each state.",
            "Bar plot of pork exports for each state.",
            "Histogram of beef exports distribution.",
            "Boxplot of wheat exports for each state.",
            "Hexagonal bin plot for the relationship between beef and pork exports.",
            "Pie chart of the share of dairy exports by state.",
            "Bar plot comparing cotton exports across states.",
            "Hexagonal bin plot illustrating the correlation between fresh and processed fruit exports.",
            "Boxplot showcasing the range and outliers of total exports for the southern states.",
            "Area plot stacked by category showing the trend of exports for a selected group of states.",
            "Hexagonal bin plot contrasting the exports of fresh fruits against dairy products.",
        ]

        hard_descriptions = [
            "Dairy earnings for the top 5 states.",
            "Meat revenue breakdown for New York state.",
            "Proportion of the export market for fresh versus processed fruits.",
            "Relationship between beef revenue and poultry revenue among states: Texas, California, Florida, New York, and Illinois.",
            "Proportion of meat revenue components for Texas."
        ]
    elif "US-shooting-incidents.csv" in dataset:
        easy_descriptions = [
            "Bar plot showing the number of incidents per state.",
            "Line plot showing the trend of incidents over the years.",
            "Histogram of the number of incidents per year.",
            "Pie chart representing the distribution of causes of incidents.",
            "Scatter plot of latitude versus longitude to visualize the locations of incidents.",
            "Bar plot showing the number of incidents involving canines.",
            "Area plot indicating the number of incidents over the years.",
            "Boxplot of incidents per year to visualize the distribution.",
            "Histogram showing the distribution of incidents based on cause.",
            "Bar plot showing the number of incidents for each cause in short form.",
            "Scatter plot showing the distribution of incidents based on latitude and longitude.",
            "Area plot representing the number of incidents involving canines over the years.",
            "Hexagonal bin plot of latitude versus longitude to visualize incident density.",
            "Pie chart representing the distribution of incidents based on whether it involved a canine or not.",
            "Histogram showing the distribution of incidents based on state.",
            "Line plot indicating the trend of incidents based on cause over the years.",
            "Scatter plot showing the distribution of incidents in different states based on latitude and longitude.",
            "Boxplot showing the distribution of incidents based on cause.",
            "Area plot representing the number of incidents in each state over the years.",
        ]
        hard_descriptions = [
            "Yearly trend of incidents.",
            "Top 3 departments with the highest incidents in 2001.",
            "Proportion of incidents by top 5 casuse of death.",
            "Top 5 casuse of death in 2001.",
            "Cumulative incidents from 2000 to 2005.",
        ]
    elif "titanic.csv" in dataset:
        easy_descriptions = [
            "Bar plot showing the number of passengers in each class.",
            "Histogram showcasing the age distribution of passengers.",
            "Pie chart representing the gender distribution aboard the Titanic.",
            "Bar plot indicating the number of survivors and non-survivors.",
            "Area plot illustrating the fare distribution over different passenger classes.",
            "Scatter plot of age against fare to see if there's a correlation.",
            "Bar plot showing the number of passengers boarding from each embarkation port.",
            "Histogram displaying the distribution of fares paid by passengers.",
            "Pie chart showcasing the distribution of passengers in lifeboats.",
            "Bar plot indicating the number of siblings/spouses each passenger had aboard.",
            "Boxplot showing the fare distribution for male and female passengers.",
            "Area plot representing the age distribution of survivors and non-survivors.",
            "Scatter plot of siblings/spouses against parents/children to see family size.",
            "Bar plot showcasing the number of passengers with and without cabins.",
            "Pie chart representing the distribution of passengers based on their embarkation port.",
            "Bar plot showing the survival rate for each gender.",
            "Boxplot displaying the age distribution for survivors and non-survivors.",
            "Area plot illustrating the fare distribution over different embarkation ports.",
            "Scatter plot of age against the number of siblings/spouses.",
            "Bar plot indicating the number of parents/children each passenger had aboard.",
            "Histogram showcasing the age distribution of male and female passengers.",
            "Pie chart representing the survival rate for each passenger class.",
            "Bar plot showing the number of passengers in each lifeboat."
        ]

        hard_descriptions = [
            "Distribution of age for survivors and non-survivors.",
            "Trend of average fare over age groups.",
            "Number of survivors from each embarkation port.",
            "Fare variability across ticket classes.",
            "Cumulative number of passengers across age groups.",
            "Proportion of passengers by ticket class."
        ]
    elif "winequality-red.csv" in dataset:
        easy_descriptions = [
            "Histogram of alcohol percentages in the wine samples.",
            "Bar plot of average salt content in different quality wines.",
            "Area plot representing the amount of residual sugar across different wine samples.",
            "Pie chart showing the proportion of wines with top 5 pH.",
            "Histogram showcasing the distribution of sulfur dioxide levels.",
            "Bar plot indicating the average citric acid content in wines of varying quality.",
            "Area plot representing the distribution of top 5 sulphate across wine samples.",
            "Pie chart illustrating the proportion of wines in each quality category.",
            "Histogram of the distribution of pH values in the wine samples.",
            "Bar plot showing average alcohol content for each wine quality score.",
            "Scatter plot comparing salt content with wine quality scores.",
            "Pie chart representing the proportion of wines with different levels of fixed acidity.",
            "Bar plot of average volatile acidity levels for different wine quality scores.",
            "Boxplot showing the range of residual sugar levels in the wine samples.",
            "Pie chart showing the proportion of wines with varying levels of citric acid.",
            "Histogram illustrating the distribution of sulphate level [0.4, 0.6]",
            "Scatter plot comparing fixed acidity 12-14 with volatile acidity [0.2, 0.4]",
            "Area plot representing the distribution of total sulfur dioxide levels [30, 50]",
            "Histogram of the distribution of density values above 1 in the wine samples",
            "Scatter plot comparing residual sugar content from 6 to 8 with alcohol percentages from 9 to 10",
        ]

        hard_descriptions = [
            "Trend of average fixed acidity levels for wines rated from 3 to 6",
            "Trend of average volatile acidity levels for wines rated from 3 to 6",
            "Distribution of density for wines rated 8.",
            "Distribution of pH levels for wines rated 8.",
            "Distribution of citric acid levels for wines rated 8.",
        ]
    elif "us-cities-top-1k.csv" in dataset:
        easy_descriptions = [
            "Bar plot of the top 5 most populous cities",
            "Area plot of populations for the top 5 cities",
            "Scatter plot of latitude versus longitude for top 5 most populous cities",
            "Pie chart showing the population distribution of the top 5 cities",
            "Bar plot of the number of cities in each state",
            "Histogram of latitudes of the top 5 most populous cities",
            "Scatter plot of population versus latitude for the top 10 most populous cities",
            "Hexagonal bin plot of latitude versus longitude for the top 5 most populous cities",
            "Bar plot showing populations of the top 5 most populous cities in California",
            "Boxplot of populations for the top 5 most populous cities in Texas",
            "Area plot of populations for the top 5 most populous cities in Florida",
            "Pie chart of the number of cities in the top 5 states",
            "Bar plot of the least populous 8 cities",
            "Boxplot of latitudes for cities in New York state",
            "Area plot of populations for the bottom 10 cities",
            "Hexagonal bin plot of population versus latitude for the top 5 most populous cities",
            "Mapbox density map showing the top 10 most populous cities"
            "Mapbox density map showing the top 5 most populous cities in California",
            "Pie chart showing the population distribution of most populous city in each of the top 5 states",
        ]

        hard_descriptions = [
            "Top 10 cities by population",
            "Population distribution of cities in Oklahoma",
            "Relationship between latitude and population for top 10 cities by population",
            "Cities with a population greater than 1 million",
            "Population comparison of cities in South Carolina and Nevada"
        ]
    elif "european_turnout.csv" in dataset:
        easy_descriptions = [
            "Histogram of the distribution of national election turnouts for Central/Eastern region countries.",
            "Boxplot of the variation in European election turnouts for Western region countries.",
            "Pie plot of the proportion of population for Central/Eastern region countries.",
            "Mapbox density plot across Mediterranean region based on European election turnouts.",
            "Pie chart of populations in Central/Eastern region countries."
        ]
        hard_descriptions = [
            "National turnouts comparison for Western region countries.",
            "Countries with a greater population than Austria's.",
            "Relationship between population size and voter turnout for Central/Eastern countries.",
            "Number of countries breakdown by region."
        ]
    elif "mpg.csv" in dataset:
        easy_descriptions = [
            "Line plot of the progression of average miles per gallon across distinct model years",
            "Bar chart comparing the average horsepower across unique cylinder configurations.",
            "Scatter plot of the relationship between a vehicle's weight and its miles per gallon for the model year 80.",
            "Pie plot representing the proportion of vehicles based on their horsepower, focusing on the top 5 horsepower values.",
            "Boxplot of the distribution of acceleration values for the vehicles with the model year 70.",
            "Area plot of the count of cars across model years.",
            "Hexagonal bin plot of cylinders versus acceleration for vehicles with 25 miles per gallon.",
        ]
        hard_descriptions = [
            "Proportion of vehicles by cylinder configurations.",
            "Trend of average vehicle weight over distinct model years.",
            "Relationship between vehicles' miles per gallon and their acceleration capabilities for the model year 80.",
        ]
    elif "gapminder2007.csv" in dataset:
        easy_descriptions = [
            "Pie plot of number of countries breakdown by continent.",
            "Scatter plot of population versus GDP in 10 African nations with the lowest GDP.",
            "Scatter plot of life expectancy versus population for 10 European countries with the highest GDP.",
            "Histogram of GDP values in 10,000 intervals for Asian countries.",
            "Box plot of life expectancy distribution for Asian countries.",
            "Bar plot of population of top 5 European nations.",
        ]
        hard_descriptions = [
            "Population comparison for the 5 most populous countries",
            "Population breakdown by continent.",
            "Top eight countries by GDP.",
            "Trend of average life expectancy across continents",
            "Relationship between GDP and life expectancy in 5 Americas countries with the lowest GDP",
            "Density of European countries based on GDP and population"
        ]
    elif "medicare.csv" in dataset:
        easy_descriptions = [
            "Bar plot of average total payments for California's top 8 hospitals by average total payments.",
            "Histogram showing the distribution of reimbursement rates for California hospitals.",
            "Boxplot of average total payments across California hospitals.",
            "Pie plot of hospital proportions in the top 5 states by average total payments.",
            "Bar chart of medical record counts in California vs. Texas.",
            "Histogram showing the distribution of number of providers across California cities.",
            "Boxplot of average covered charges across Texas hospitals",
        ]
        hard_descriptions = [
            "Average total payment comparison across Wisconsin hospitals.",
            "Breakdown of record count by medical classification.",
            "List provider names with reimbursement rate over 1.",
            "Proportion of 'Alcohol and Drug Use' records in top 5 states, based on the number of hospitals.",
            "Number of Alcohol and Drug Use records comparison in California vs. Texas."
        ]
    elif "volcano_db.csv" in dataset:
        easy_descriptions = [
            "Bar plot of the 5 countries with the most volcanoes listed.",
            "Line plot showing the elevation trend of the 8 highest volcanoes in Japan.",
            "Histogram of the distribution of volcano elevations for volcanoes in China.",
            "Boxplot of the elevation distribution of volcanoes in the United States.",
            "Area plot of the elevation progression of the 8 highest volcanoes in Russia.",
            "Scatter plot of latitude versus longitude for volcanoes in Peru.",
            "Mapbox density map of the concentration of volcanoes in Mexico.",
            "Pie plot of the distribution of volcano types in China.",
            "Pie plot of the distribution of volcano status.",
        ]
        hard_descriptions = [
            "Breakdown of volcano types in the United States.",
            "Distribution of volcano elevations in Mexico.",
            "Relationship between latitude and elevation for the 8 southernmost volcanoes.",
            "Comparison of the number of volcanoes in the top 5 countries.",
            "Trend of volcano elevations in the 5 highest volcanoes in Mexico.",
            "Volcanoes with elevations above 4000.",
            "Breakdown of volcano status in the United States.",
        ]
    else:
        raise ValueError("No automation of test cases curation for the given dataset.")

    combined_list = []
    for desc in easy_descriptions:
        item = {
            "uuid": generate_id(dataset, desc),
            "dataset": dataset,
            "language": "English",
            "complexity": "easy",
            "description": desc
        }
        combined_list.append(item)

    for desc in hard_descriptions:
        item = {
            "uuid": generate_id(dataset, desc),
            "dataset": dataset,
            "language": "English",
            "complexity": "hard",
            "description": desc
        }
        combined_list.append(item)

    test_case_path = f'data/{mode}/{mode}.json'

    # Convert the combined list to JSON
    with open(test_case_path, 'w') as file:
        json.dump(combined_list, file, indent=4)


def gen_golden(dataset, mode="train"):
    """
    Generate:
        - code that generates golden plots -> train-code.json
        - golden plots in json -> train-golden.json
    """
    spark = SparkSession.builder.getOrCreate()
    spark_ai = SparkAI(spark_session=spark, verbose=True)
    spark_ai.activate()

    if "volcano_db.csv" in dataset:
        pdf = pd.read_csv(dataset, encoding='ISO-8859-1')
    else:
        pdf = pd.read_csv(dataset)
    df = spark_ai._spark.createDataFrame(pdf)

    include_keys = INCLUDE_KEYS
    golden_codes = []
    golden_jsons = []

    code_path = f'data/{mode}/{mode}-code.json'
    golden_path = f'data/{mode}/{mode}-golden.json'
    test_case_path = f'data/{mode}/{mode}.json'

    with open(test_case_path, 'r') as file:
        test_cases = json.load(file)

    for test_case in test_cases:
        if test_case['dataset'] == dataset:
            uuid = test_case['uuid']
            try:
                original_code = df.ai.plot(test_case['description'])
                code = substitute_show_to_json(original_code)

                golden_codes.append({
                    'uuid': uuid,
                    'code': code
                })

                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    exec(compile(code, "plot_df-CodeGen-benchmark", "exec"))
                captured_output = buffer.getvalue()[:-1]

                # Take data field (not layout field) as golden
                golden_jsons.append({
                    'uuid': uuid,
                    'plot_json': filter_golden_json_data(captured_output, include_keys)
                })
            except Exception as e:
                print(f"Test case with UUID {uuid} failed due to: {str(e)}")
                continue

    # Convert the golden_codes list to JSON
    with open(code_path, 'w') as file:
        json.dump(golden_codes, file, indent=4)

    # Convert the golden_jsons list to JSON
    with open(golden_path, 'w') as file:
        json.dump(golden_jsons, file, indent=4)


if __name__ == "__main__":
    # for dataset in TRAIN_DATASETS:
    #     gen_test_case(dataset)
    #     gen_golden(dataset)
    # for dataset in TEST_DATASETS:
    #     gen_test_case(dataset, mode="test")
    #     gen_golden(dataset, mode="test")
    pass
