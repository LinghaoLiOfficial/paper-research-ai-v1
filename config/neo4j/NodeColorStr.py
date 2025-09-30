from config.neo4j.NodeLabelStr import NodeLabelStr


# 节点颜色
class NodeColorStr:
    USER = "#c990c0"
    PDF_FILE = "#f79767"
    TOPIC_TEXT = "#57c7e3"
    SENTENCE_TEXT = "#f16667"
    RECORD_TEXT = "#d9c8ae"

    ALL = "#a5abb6"

    A = "#8dcc93"
    B = "#ecb5c9"
    C = "#4c8eda"
    D = "#ffc454"
    E = "#da7194"
    F = "#569480"

    @classmethod
    def mapping(cls):
        return {
            NodeLabelStr.USER: cls.USER,
            NodeLabelStr.PDF_FILE: cls.PDF_FILE,
            NodeLabelStr.TOPIC_TEXT: cls.TOPIC_TEXT,
            NodeLabelStr.SENTENCE_TEXT: cls.SENTENCE_TEXT,
            NodeLabelStr.RECORD_TEXT: cls.RECORD_TEXT,

            NodeLabelStr.ALL: cls.ALL
        }
