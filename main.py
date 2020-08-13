# library imports
import numpy as np
import scipy.stats
import pandas as pd

# project imports
from ml.random_forest import RandomForest
from graphs.PlotManger import PlotManager
from utils.io.path_handler import PathHandler
from ml.prepare_ml_dataset import PrepareMlDataset

# ML libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


class Main:
    """
    Single file with the main logic, in progress splitting of logic to other files
    """

    @staticmethod
    def already_run_analysis():
        # read data
        merged_df = Main.read_data_to_framework(data_path=PathHandler.get_relative_path_from_project_inner_folders(["data", "students_with_tests.xlsx"]),
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

        # find final exam score
        final_scores = Main._final_exam(df=merged_df)

        # show score distribution
        PlotManager.show_distribution(data=final_scores,
                                      name="final_scores",
                                      title="final score distribution",
                                      x_axis="score",
                                      y_axis="count")

        mean_final_scores = np.average(final_scores)
        std_final_scores = np.std(final_scores)
        with open("answers/final_score_stats.txt", "w") as final_score_file:
            final_score_file.write("Mean: {:.2f}\nStd: {:.2f}".format(mean_final_scores, std_final_scores))

        # check first hypothesis
        Main.good_students_prefer_recorded_lectures(df=merged_df)

        # load second part of data
        questioner = Main.read_data_to_framework(data_path=PathHandler.get_relative_path_from_project_inner_folders(["data", "questioner_fixed.xlsx"]),
                                                 sheet_name="Form Responses 1")

        Main.common_analysis(df=questioner)

    @staticmethod
    def good_students_prefer_recorded_lectures(df):
        # find final exam score
        final_scores = Main._final_exam(df=df)

        # calc stats on col
        mean_final_scores = np.average(final_scores)
        std_final_scores = np.std(final_scores)
        threshold = mean_final_scores + 0.5 * std_final_scores

        # add to df
        df["final_score"] = final_scores

        good_students = df.loc[df['final_score'] >= threshold]
        bad_students = df.loc[df['final_score'] < threshold]

        guess_col_names = ["read_slides_happiness_prefer_online",
                           "read_slides_happiness_prefer_recorded_online",
                           "online_cannot_replace_frontal",
                           "recorded_lectures_i_can_rewatch_unclear_sections",
                           "recorded_lectures_watching_speed",
                           "recorded_lectures_skipable",
                           "recorded_lectures_similar_questions_to_mine",
                           "recorded_lectures_tend_to_delay_watch",
                           "recorded_lectures_cannot_ask_questions_problem",
                           "recorded_lectures_skip_additional_explanation",
                           "recorded_lectures_tend_to_watch_higher_speed"]
        with open("answers/good_students_prefer_recorded_lectures.txt", "w") as final_score_file:
            final_score_file.write("Good Students: {} | Bad students: {} | threshold: {:.2f}\n\n".format(good_students.shape[0], bad_students.shape[0], threshold))
            # test few coloms
            for col_name in guess_col_names:
                t_test_score, p_value = scipy.stats.ttest_ind(good_students[col_name], bad_students[col_name])
                final_score_file.write("{}: t-score: {:.3f} (p = {:.5f})\n".format(col_name, t_test_score, p_value))

    @staticmethod
    def common_analysis(df):

        PlotManager.show_distribution(df["age"],
                                      hist=True,
                                      kde=True,
                                      bins=max(df["age"]),
                                      name="common_analysis_age",
                                      title="Age Distrebution",
                                      x_axis="Age [year]",
                                      y_axis="Density")
        total = df.count()[0]
        male = df[df["gender"] == 1].count()[0]
        PlotManager.pie_chart(sizes=[male / total, 1 - (male / total)],
                              labels=["Male", "Female"],
                              name="gender")

        PlotManager.pie_chart(sizes=[df[df["faculty"] == type].count()[0] / total for type in df.faculty.unique()],
                              labels=df.faculty.unique(),
                              name="faculty")

    @staticmethod
    def predict_rf_final_score(df,
                               debug: bool = False,
                               max_depth: int = 5):
        # find final exam score
        final_scores = Main._final_exam(df=df)
        # add to df
        df["final_score"] = final_scores

        # keep only the wanted coloums
        print("cols = {} (size = {})".format(PrepareMlDataset.dt_cols, len(PrepareMlDataset.dt_cols)))
        df = df[PrepareMlDataset.dt_cols]
        df = PrepareMlDataset.fix_pysicometry(df=df)
        df = PrepareMlDataset.normalize(df=df,
                                        feature="final_score",
                                        factor=100)

        y_cols = ["final_score"]
        x, y = PrepareMlDataset.get_xs_ys(df=df, y_cols=y_cols)
        x_train, y_train, x_test, y_test = PrepareMlDataset.split_data(x, y, 0.3)
        model = RandomForest()
        model.train(x=x_train,
                    y=y_train,
                    type=RandomForest.single_tree,
                    max_depth=max_depth)
        test_score = model.manual_test(x_test=x_test,
                                       y_test=y_test,
                                       factor=100,
                                       is_abs_error=True,
                                       debug=debug)
        with open("answers/rf_model_predict_final_score_max_depth_{}.txt".format(max_depth), "w") as result_file:
            result_file.write("Train: {} samples\nTest: {} samples with {:.3f} acc\n".format(x_train.shape[0], x_test.shape[0], test_score))

        df = df.drop(columns=y_cols)
        model.export_graph(feature_names=list(df.columns),
                           print_text="final_score_features",
                           print_to_console=True)

        model.print_importance(name="final_score_features",
                               ordered=False)

    @staticmethod
    def predict_rf_final_score_k_fold_test(df,
                                           debug: bool = False,
                                           max_depth: int = 5,
                                           k: int = 100):
        # find final exam score
        final_scores = Main._final_exam(df=df)
        # add to df
        df["final_score"] = final_scores

        # keep only the wanted coloums
        df = df[PrepareMlDataset.dt_cols]
        df = PrepareMlDataset.fix_pysicometry(df=df)
        df = PrepareMlDataset.normalize(df=df,
                                        feature="final_score",
                                        factor=100)
        x, y = PrepareMlDataset.get_xs_ys(df=df, y_cols=["final_score"])

        test_scores = []
        for i in range(k):
            x_train, y_train, x_test, y_test = PrepareMlDataset.split_data(x, y, 0.3)
            model = RandomForest()
            model.train(x=x_train,
                        y=y_train,
                        type=RandomForest.single_tree,
                        max_depth=max_depth)
            test_score = model.manual_test(x_test=x_test,
                                           y_test=y_test,
                                           is_abs_error=True,
                                           factor=100,
                                           debug=debug)
            test_scores.append(test_score)
        print("\nAvg test score: {:.4f}\n".format(sum(test_scores) / len(test_scores)))
        with open("answers/rf_model_predict_final_score_k_fold_test_max_depth_{}.txt".format(max_depth), "w") as result_file:
            result_file.write("k = {}, average accuracy = {:.4f}".format(k, sum(test_scores) / len(test_scores)))

    @staticmethod
    def study_method_does_not_change_score(df):

        # find final exam score
        partial_final_scores = Main._partial_final_exam(df=df)

        # add to df
        df["partial_final_scores"] = partial_final_scores

        # get the 3 groups
        recorded_students = df.loc[df["last_lession"] == 1]
        live_later_students = df.loc[df["last_lession"] == 2]
        live_students = df.loc[df["last_lession"] == 3]

        students_groups = [recorded_students, live_later_students, live_students]
        names = ["recorded_students", "live_later_students", "live_students"]

        with open("answers/study_method_does_not_change_score.txt", "w") as final_score_file:
            # test few coloms
            for first_group_index in range(len(students_groups)):
                for second_group_index in range(first_group_index + 1, len(students_groups)):
                    t_test_score, p_value = scipy.stats.ttest_ind(students_groups[first_group_index]["partial_final_scores"],
                                                                  students_groups[second_group_index]["partial_final_scores"])
                    final_score_file.write("{} X {} : t-score: {:.3f} (p = {:.5f})\n".format(names[first_group_index],
                                                                                             names[second_group_index],
                                                                                             t_test_score,
                                                                                             p_value))

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
    def _final_exam(df) -> list:
        """
        :param df: the data frame with the data
        :return: list of coronations
        """
        col_names = list(df.columns)
        needed_col_names = ["exam_qs_1", "exam_qs_1_bonous", "exam_qs_2", "exam_qs_3", "exam_qs_4", "exam_qs_5", "exam_qs_6a", "exam_qs_6b"]
        columns_index = [index for index, name in enumerate(col_names) if name in needed_col_names]
        return df.iloc[:, columns_index].sum(axis=1)

    @staticmethod
    def _partial_final_exam(df) -> list:
        """
        :param df: the data frame with the data
        :return: list of coronations
        """
        col_names = list(df.columns)
        needed_col_names = ["exam_qs_1", "exam_qs_1_bonous", "exam_qs_2", "exam_qs_3", "exam_qs_4", "exam_qs_5"]
        columns_index = [index for index, name in enumerate(col_names) if name in needed_col_names]
        return df.iloc[:, columns_index].sum(axis=1)

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

    @staticmethod
    def run_analysis():
        df = Main.read_data_to_framework(data_path=PathHandler.get_relative_path_from_project_inner_folders(["data", "students_with_tests.xlsx"]), sheet_name="Sheet1")
        df = df[df["pysicometry"] < 200] # clear bad lines

        # add final score
        col_names = list(df.columns)
        needed_col_names = ["exam_qs_1", "exam_qs_1_bonous", "exam_qs_2", "exam_qs_3", "exam_qs_4", "exam_qs_5"]
        columns_index = [index for index, name in enumerate(col_names) if name in needed_col_names]
        df["final_score"] = df.iloc[:, columns_index].sum(axis=1)

        # find good students threshold
        scores = list(df["final_score"])
        mean_score = sum(scores) / len(scores)
        std_score = np.std(scores)
        upper_score = mean_score + 0.5 * std_score

        interestring_coloums = ["bagrot", "first_semester_score", "pysicometry", "prefer_lecture_and_practice",
                                "prefer_record_lecture_and_not_practice", "prefer_not_lecture_and_recorded_practice",
                                "prefer_lecture_and_practice",	"prefer_lecture_and_frontal_practice",
                                "read_slides_happiness", "study_just_before_the_exam", "hw_each_week"]

        # split to groups
        last_lession = {}
        for i in range(3):
            key = "group_{}".format(i + 1)
            last_lession[key] = (df[df["last_lession"] == i + 1])
            last_lession[key]["final_score"] = last_lession[key]["final_score"].apply(lambda x: 1 if x > upper_score else 0) # fix the 'y' value

            X = last_lession[key][interestring_coloums]
            y = last_lession[key]["final_score"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
            gnb = GaussianNB()
            gnb.fit(X_train, y_train)

            y_pred = gnb.predict(X_test)
            print("From {} tests: {:.2f}% passed".format(X_test.shape[0], 100 * (y_test == y_pred).sum() / X_test.shape[0]))

            with open("{}_naive_bayes.txt".format(key), "w") as answer_file:
                answer_string = "features = {}\n\nSigma = {}\nTheta = {}".format(interestring_coloums, gnb.sigma_, gnb.theta_)
                answer_file.write(answer_string)
                print("\n\nFor case {}:\n{}".format(key, answer_string))


if __name__ == '__main__':
    Main.run_analysis()
