import numpy as np
from sklearn.model_selection import train_test_split


class PrepareMlDataset:
    """
    Static class to handle data preparation for ML tasks
    """

    dt_cols = list(set(["bagrot", "first_semester_score", "pysicometry", "bagrot_points", "last_lession",
                        "recorded_lectures_watching_speed", "recorded_lectures_skipable", "recorded_lectures_tend_to_delay_watch",
                        "recorded_lectures_skip_additional_explanation", "recorded_lectures_tend_to_watch_higher_speed",
                        "study_just_before_the_exam", "hw_each_week", "highly_motivated_for_learning", "care_only_for_final_grade",
                        "summarizing_during_lecture", "final_score"]))

    @staticmethod
    def fix_pysicometry(df):
        df = df[df["pysicometry"] > 0]
        df = df[df["pysicometry"] < 200]
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def normalize(df, feature, factor: float = 100):
        max_val = df[feature].max()
        min_val = df[feature].min()
        df[feature] = df[feature].map(lambda x: (x - min_val) / (max_val - min_val) * factor)
        return df

    @staticmethod
    def get_xs_ys(df, y_cols: list):
        # retrieve y values
        y_s = df[y_cols].values
        df = df.drop(columns=y_cols)
        x_s = df.values
        return x_s, y_s

    @staticmethod
    def split_data(x_s, y_s, test_percent: float = 0.3):
        # convert values to int
        y_s = np.asarray([tuple(y.astype(int)) for y in y_s])
        # split data to train, test, validation using 70%, 15%, 15%
        x_train, x_test, y_train, y_test= train_test_split(x_s, y_s, test_size=test_percent)
        return x_train, y_train, x_test, y_test



