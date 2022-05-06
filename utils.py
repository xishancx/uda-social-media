from settings import flags
import pandas as pd
import numpy as np



def get_input_df(input_path = flags['input_path']):
    """
    :param input_path:
    input_path: Path for dataset file
    :return:
    data_df: Extracted data from CSV file
    """
    data_df = pd.read_csv(input_path, encoding="ISO-8859-1")

    # Extract and re-order relevant data from csv
    data_df = data_df.iloc[:, [0, 1, 5]]
    data_df.columns = ['Target', 'ID', 'Input']
    data_df = data_df[['ID', 'Input', 'Target']]
    data_df.Target = data_df.Target.replace(4, 1)

    return data_df


