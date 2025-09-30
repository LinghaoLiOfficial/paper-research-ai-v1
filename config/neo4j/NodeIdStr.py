from config.neo4j.NodeLabelStr import NodeLabelStr


# 节点ID英文名
class NodeIdStr:
    USER = "user_id"
    PDF_FILE = "file_id"
    TOPIC_TEXT = "text_id"
    SENTENCE_TEXT = "text_id"
    RECORD_TEXT = "text_id"

    @classmethod
    def mapping(cls):
        return {
            NodeLabelStr.USER: cls.USER,
            NodeLabelStr.PDF_FILE: cls.PDF_FILE,
            NodeLabelStr.TOPIC_TEXT: cls.TOPIC_TEXT,
            NodeLabelStr.SENTENCE_TEXT: cls.SENTENCE_TEXT,
            NodeLabelStr.RECORD_TEXT: cls.RECORD_TEXT
        }
