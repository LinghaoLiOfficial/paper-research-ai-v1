from config.neo4j.NodeLabelStr import NodeLabelStr


# 节点边缘颜色
class NodeEdgeColorStr:
    USER = "#b261a5"
    PDF_FILE = "#f36924"
    TOPIC_TEXT = "#23b3d7"
    SENTENCE_TEXT = "#eb2728"
    RECORD_TEXT = "#c0a378"

    A = "#5db665"
    B = "#da7298"
    C = "#2870c2"
    D = "#d7a013"
    E = "#cc3c6c"
    F = "#447666"

    @classmethod
    def mapping(cls):
        return {
            NodeLabelStr.USER: cls.USER,
            NodeLabelStr.PDF_FILE: cls.PDF_FILE,
            NodeLabelStr.TOPIC_TEXT: cls.TOPIC_TEXT,
            NodeLabelStr.SENTENCE_TEXT: cls.SENTENCE_TEXT,
            NodeLabelStr.RECORD_TEXT: cls.RECORD_TEXT
        }
