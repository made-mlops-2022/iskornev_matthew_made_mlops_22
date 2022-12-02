from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network
import pandas as pd
import matplotlib.pyplot as plt
import click
import os
from pathlib import Path


TARGET = 'condition'


@click.command("download")
@click.option('--output_dir', '-od',
              default='/data/raw/',
              help='Please enter path to output data. Default path - /data/raw/')
def get_data(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    synthetic_data, description_file = get_synth_data(output_dir)
    synthetic_df = pd.read_csv(synthetic_data)
    df = pd.read_csv('heart_cleveland_upload.csv')
    attribute_description = read_json_file(description_file)['attribute_description']
    inspector = ModelInspector(df, synthetic_df, attribute_description)
    inspector.mutual_information_heatmap()
    plt.savefig('heatmap.png')

    X = synthetic_df.drop(columns=TARGET)
    target = synthetic_df[TARGET]
    os.remove(synthetic_data)
    X.to_csv(Path(output_dir).joinpath('data.csv'), index=False)
    target.to_csv(Path(output_dir).joinpath('target.csv'), index=False)


def get_synth_data(output_dir: str) -> tuple[Path, str]:
    input_data = 'heart_cleveland_upload.csv'
    description_file = 'description.json'

    synthetic_data = Path(output_dir).joinpath('synthetic_data.csv')

    categorical_attributes = {'sex': True, 'cp': True, 'fbs': True, 'restecg': True,
                              'exang': True, 'slope': True, 'ca': True, 'thal': True}

    threshold_value = 7
    epsilon = 1
    degree_of_bayesian_network = 2
    num_tuples_to_generate = 400

    describer = DataDescriber(category_threshold=threshold_value)
    describer.describe_dataset_in_correlated_attribute_mode(
        dataset_file=input_data, epsilon=epsilon, k=degree_of_bayesian_network,
        attribute_to_is_categorical=categorical_attributes
    )

    describer.save_dataset_description_to_file(description_file)
    display_bayesian_network(describer.bayesian_network)
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
    generator.save_synthetic_data(synthetic_data)
    return synthetic_data, description_file


if __name__ == '__main__':
    get_data()
