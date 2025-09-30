from config.neo4j.NodeLabelStr import NodeLabelStr
from config.neo4j.NodeNameStr import NodeNameStr


# 节点类型英文名对应节点名称英文名
class NodeSearchStr:
    @classmethod
    def get(cls):
        return {
            NodeLabelStr.USER: NodeNameStr.USER,
            NodeLabelStr.PDF_FILE: NodeNameStr.PDF_FILE,
            NodeLabelStr.TOPIC_TEXT: NodeNameStr.TOPIC_TEXT,
            NodeLabelStr.SENTENCE_TEXT: NodeNameStr.SENTENCE_TEXT,
            NodeLabelStr.RECORD_TEXT: NodeNameStr.SENTENCE_TEXT
        }