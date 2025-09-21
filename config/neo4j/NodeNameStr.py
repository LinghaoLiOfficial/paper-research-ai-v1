from config.neo4j.NodeLabelStr import NodeLabelStr


# 节点名称英文名
class NodeNameStr:
    USER = "username"
    PDF_FILE = "file_name"
    TOPIC_TEXT = "text_name"
    SENTENCE_TEXT = "text_name"
    RECORD_TEXT = "text_name"

    @classmethod
    def mapping(cls):
        return {
            NodeLabelStr.USER: cls.USER,
            NodeLabelStr.PDF_FILE: cls.PDF_FILE,
            NodeLabelStr.TOPIC_TEXT: cls.TOPIC_TEXT,
            NodeLabelStr.SENTENCE_TEXT: cls.SENTENCE_TEXT,
            NodeLabelStr.RECORD_TEXT: cls.RECORD_TEXT
        }
