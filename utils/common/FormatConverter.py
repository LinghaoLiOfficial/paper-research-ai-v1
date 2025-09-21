class FormatConverter:

    @classmethod
    def list_objects_to_dict(cls, list: list):
        return [x.to_dict() for x in list]

    @classmethod
    def object_list_to_table(cls, list: list):
        if len(list) == 0:
            return {
                "head": [],
                "body": [[]]
            }

        return {
            "head": [x for x in list[0].keys()],
            "body": [[y for y in x.values()] for x in list]
        }

