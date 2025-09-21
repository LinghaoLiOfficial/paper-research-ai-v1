import os


class LocalDataFileReader:

    @classmethod
    def read_folder_name_list(cls):
        folder_name_list = [
            {
                "id": 0,
                "name": "内蒙古风场-风机时间序列数据",
                "value": "su_you"
            },
            {
                "id": 1,
                "name": "湘电风场-风机时间序列数据",
                "value": "xiang_dian"
            },
        ]

        return folder_name_list

    @classmethod
    def read_file_name_dict(cls):
        base_path = "./data/{}/{}"

        file_name = "su_you"

        time_scale = "time_scale_1"

        path_list = [base_path.format(file_name, time_scale)]

        data_file_name_dict = {}
        for path in path_list:

            data_file_name_list = []
            for i, sub_path in enumerate(os.listdir(path)):
                data_file_name_list.append({
                    "id": i,
                    "name": sub_path,
                    "value": sub_path
                })

            data_file_name_list.append({
                "id": len(data_file_name_list),
                "name": "全部"
            })

            data_file_name_dict[file_name] = data_file_name_list

        return data_file_name_dict
