from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network
import pandas as pd
import matplotlib.pyplot as plt


def main():
    input_data = ('data/raw/heart_cleveland_upload.csv')
    # mode = 'correlated_attribute_mode'
    description_file = 'tests/synth_data/description.json'
    synthetic_data = 'tests/synth_data/synthetic_data.csv'

    categorical_attributes = {'sex': True, 'cp': True, 'fbs': True, 'restecg': True,
                              'exang': True, 'slope': True, 'ca': True, 'thal': True}

    threshold_value = 7
    epsilon = 1
    degree_of_bayesian_network = 2
    num_tuples_to_generate = 100

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

    synthetic_df = pd.read_csv(synthetic_data)
    df = pd.read_csv('data/raw/heart_cleveland_upload.csv')

    attribute_description = read_json_file(description_file)['attribute_description']
    inspector = ModelInspector(df, synthetic_df, attribute_description)
    inspector.mutual_information_heatmap()
    plt.savefig('tests/synth_data/heatmap.png')


if __name__ == '__main__':
    main()
