import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from entity.common.StrucResult import StrucResult


class FileSaver:

    BASE_PATH = "./storage/{}"

    @classmethod
    def get_file_path(cls, uuid_file_name):
        return cls.BASE_PATH.format(uuid_file_name)

    @classmethod
    def save(cls, file, uuid_file_name):

        try:
            file_path = cls.get_file_path(uuid_file_name)

            file.save_into_database(file_path)

        except Exception as e:
            print(e)
            return StrucResult.build_error()

        return StrucResult.build_success()


    @classmethod
    def pyarrow_save_file(cls, df: pd.DataFrame):
        # TODO
        # 用arrow格式保存
        pq.write_table(pa.Table.from_pandas(df), "df_parquet.parquet")

    @classmethod
    def pyarrow_load_file(cls):
        # TODO
        # 用arrow格式保存
        df = pq.read_table("df_parquet.parquet").to_pandas()

        return df
