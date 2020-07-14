# library imports
import numpy as np
import scipy.stats
import pandas as pd

# project imports
from graphs.PlotManger import PlotManager
from utils.io.path_handler import PathHandler


class Main:
    """
    Single file with the main logic, in progress splitting of logic to other files
    """

    @staticmethod
    def run_analysis():
        # read data
        merged_df = Main.read_data_to_framework(data_path=PathHandler.get_relative_path_from_project_inner_folders(["data", "single_sheet_data.xlsx"]),
                                                sheet_name="Sheet1")

        # generate corr matrix
        PlotManager.corolation_matrix(df=merged_df)

        # first test
        pass_value = 0.4
        corolations = Main._brute_force_pearson_two_col(df=merged_df, pass_value=pass_value, pass_p_value=0.1)
        with open("answers/coronations_{}.txt".format(pass_value), "w", encoding="utf-8") as corolation_file:
            for corolation in corolations:
                corolation_file.write("Coronation between '{}' and '{}' with score {:.3f} (p={:.3f})\n".format(corolation[0],
                                                                                                               corolation[1],
                                                                                                               corolation[2],
                                                                                                               corolation[3]))

    @staticmethod
    def smart_pearson(df,
                      columns_index: list,
                      weights: list,
                      target_column_index: int):
        """
        Returns the pearson correlation between a column and a linear combination of columns.

        Parameters:
            data: The Pandas dataframe to take the values from.
            columns_index: a list of indices of columns from the dataframe to be multiplied by the provided weights.
            weights: a list of weights to be multiplied by the column corresponding to the provided indices.
            target_column_index: the index of the additional column to calculate the pearson correlation.

        Returns:
            the pearson correlation between:
            1. data[target_column_index], and
            2. the sum of data[columns_index[i]*weights[i]] for every i in 0 ... len(columns_index), which has to be equal to len(weights).
        """

        # make sure the arguments will work
        assert len(columns_index) == len(weights)
        assert sum(weights) == 1

        # When doing the linear combination, NaN values are ignored. That is, they are taken as 0.
        # Therefore, even though Pandas corr() function automatically removes rows with some NaN value,
        # we need to remove in advance the rows with a NaN value for one of the columns_index.

        clean_df = df.dropna(subset=df.columns[columns_index])
        columns_linear_comb = (clean_df.iloc[:, columns_index] * weights).sum(axis=1)
        target_column = clean_df.iloc[:, target_column_index]
        return scipy.stats.pearsonr(columns_linear_comb, target_column)

    @staticmethod
    def _brute_force_pearson_two_col(df, pass_value: float, pass_p_value: float) -> list:
        """
        :param df: the data frame with the data
        :param pass_value: the abs value of pearson we are interested in
        :param pass_p_value: the p_value we found relevant
        :return: list of coronations
        """
        answers = []

        col_names = list(df.columns)
        for name_index_i in range(len(col_names) - 1):
            for name_index_j in range(name_index_i + 1, len(col_names)):
                try:
                    value, p_value = Main._calc_pearson_two_col(df, col_names[name_index_i], col_names[name_index_j])
                    if abs(value) > pass_value and p_value < pass_p_value:
                        answers.append([col_names[name_index_i], col_names[name_index_j], value, p_value])
                except Exception as error:
                    print("Was not able to perform on {} with {}".format(col_names[name_index_i], col_names[name_index_j]))
        return answers

    @staticmethod
    def _calc_pearson_two_col(df, col_name_1: str, col_name_2: str) -> tuple:
        """
        :param df: the data frame with the data
        :param col_name_1: the first col we are interested in
        :param col_name_2: the second col we are interested in
        :return: coronation and p-value
        """
        return scipy.stats.pearsonr(df[col_name_1], df[col_name_2])

    @staticmethod
    def one_time_prepare():
        """
        used to convert the ooriginal2 files of answers and exam scores into single file and save as new Excel file
        """
        contact_information_df = Main.read_data_to_framework(data_path=PathHandler.get_relative_path_from_project_inner_folders(["data", "whole_data.xlsx"]),
                                                             sheet_name="google_questions")
        exam_df = Main.read_data_to_framework(data_path=PathHandler.get_relative_path_from_project_inner_folders(["data", "whole_data.xlsx"]),
                                              sheet_name="exam_results")

        # show head to understand what we see in the data
        print("\n\ncontact_information_df:\n{}".format(contact_information_df.head()))
        print("Rows: {}, Cols: {}".format(contact_information_df.shape[0], contact_information_df.shape[1]))
        print("\n\nexam_df:\n{}".format(exam_df.head()))
        print("Rows: {}, Cols: {}".format(exam_df.shape[0], exam_df.shape[1]))

        # merge to one data frame
        whole_df = pd.merge(contact_information_df, exam_df, on="id")
        print("\n\nwhole_df:\n{}".format(whole_df.head()))
        print("Rows: {}, Cols: {}".format(whole_df.shape[0], whole_df.shape[1]))

        # save back
        whole_df.to_excel(PathHandler.get_relative_path_from_project_inner_folders(["data", "single_sheet_data.xlsx"]))
        print("Saved 'single_sheet_data.xlsx' in the wanted format")

    @staticmethod
    def read_data_to_framework(data_path: str, sheet_name: str):
        """ just read the data from the needed file and sheet by name """
        return pd.read_excel(data_path,
                             index_col=0,
                             sheet_name=sheet_name)


if __name__ == '__main__':
    Main.run_analysis()
