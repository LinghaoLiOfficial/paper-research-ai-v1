import pandas as pd
import os


class DataImport:

    @classmethod
    def import_multi_data(cls, root_path: str):

        all_data_path = cls.import_all_data(root_path)

        df_list = []
        for data_path in all_data_path:

            df = cls.import_data(data_path, False)

            df_list.append(df)

        new_df = pd.concat(df_list, axis=0)

        return new_df

    @classmethod
    def import_all_data(cls, root_path: str):

        selected_wt_id_list = [1, 3, 4, 5, 6, 7]

        data_path_list = ["{}/{}".format(root_path, x) for x in os.listdir(root_path) if int(x.replace("_df.csv", "")[-2:]) in selected_wt_id_list]

        return data_path_list

    @classmethod
    def import_data(cls, data_path: str, add_timestamp: bool):

        if add_timestamp:

            df = pd.read_csv(data_path)

            df["timestamp"] = df["timestamp"].apply(lambda x: x[:-1] + "0")

            wt_id = int(data_path.replace("_df.csv", "")[-2:])

            df["wt_id"] = wt_id

            return df, wt_id

        df = pd.read_csv(data_path)

        return df




