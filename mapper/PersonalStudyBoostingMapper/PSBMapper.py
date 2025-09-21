from config.neo4j.NodeIdStr import NodeIdStr
from config.neo4j.NodeLabelStr import NodeLabelStr
from config.neo4j.NodeSearchStr import NodeSearchStr
from utils.database.CypherDriver import CypherDriver
from utils.database.SqlDriver import SqlDriver


class PSBMapper:

    @classmethod
    def delete_node_from_id(cls, node_id):
        cypher = f"MATCH (n) WHERE id(n) = {node_id} DETACH DELETE n"
        return CypherDriver.execute_write(cypher)

    @classmethod
    def set_node_property_value_from_key(cls, node_id, node_key, node_value):
        cypher = f"MATCH (n) WHERE id(n) = {node_id} SET n.{node_key} = '{node_value}'"
        return CypherDriver.execute_write(cypher)

    @classmethod
    def match_all_from_all_name(cls, user_id, text_to_search):
        cypher = f"MATCH (n:{NodeLabelStr.USER})-[R*]-(N) WHERE n.user_id = '{user_id}' AND (" + " OR ".join([f"N.{prop} CONTAINS '{text_to_search}'" for prop in NodeSearchStr.get().values()]) + ") RETURN DISTINCT N AS properties, labels(N) AS labels, id(N) AS id"
        return CypherDriver.execute_read(cypher, ["properties", "labels", "id"])

    @classmethod
    def match_all_from_user_node_return_path_max_length(cls, user_id):
        cypher = f"MATCH P=(n:{NodeLabelStr.USER})-[R*]-(N) WHERE n.user_id = '{user_id}' RETURN max(length(P)) AS maxLength"
        return CypherDriver.execute_read(cypher, ["maxLength"])

    @classmethod
    def match_user_node(cls, user_id):
        cypher = f"MATCH (N:{NodeLabelStr.USER}) WHERE N.user_id = '{user_id}' RETURN DISTINCT N AS properties, labels(N) AS labels, id(N) AS id"
        return CypherDriver.execute_read(cypher, ["properties", "labels", "id"])

    @classmethod
    def match_all_from_user_node_return_mapping(cls, user_id, layer):
        cypher = f"MATCH (n0:{NodeLabelStr.USER})" + "".join([f"-[]-(n{i + 1})" for i in range(layer)]) + f" WHERE n0.user_id = '{user_id}' RETURN DISTINCT n{layer} AS properties, labels(n{layer}) AS labels, id(n{layer - 1}) AS start_id, id(n{layer}) AS end_id"
        return CypherDriver.execute_read(cypher, ["properties", "labels", "start_id", "end_id"])

    @classmethod
    def select_all_from_file_where_owner_id(cls, owner_id):
        sql = f"select * from file where owner_id = '{owner_id}'"
        return SqlDriver.execute_read(sql, File)

    @classmethod
    def match_all_from_user_node_where_user_id_and_node_label(cls, user_id, node_label):
        cypher = f"MATCH (n:{NodeLabelStr.USER})-[*0..]-(N) WHERE n.user_id = '{user_id}' and '{node_label}' in labels(N) RETURN DISTINCT N AS properties"
        return CypherDriver.execute_read(cypher, ["properties"])

    @classmethod
    def merge_text_node(cls, chosen_node_label, chosen_node_id, related_node_label, text_id, related_text_name):
        cypher = f"MATCH (n:{chosen_node_label}) WHERE n.{NodeIdStr.mapping()[chosen_node_label]} = '{chosen_node_id}' MERGE (n1:{related_node_label} {{text_id: '{text_id}', text_name: '{related_text_name}'}})-[:r]-(n)"
        return CypherDriver.execute_write(cypher)

    @classmethod
    def transaction_insert_file(cls, file_id, file_name, file_size, file_type, owner_id, input=None):
        sql = f"insert into file (file_id, file_name, file_size, file_type, owner_id) values ('{file_id}', '{file_name}', {file_size}, '{file_type}', '{owner_id}')"
        return SqlDriver.execute_transaction_write(sql, input)

    @classmethod
    def transaction_merge_file_node(cls, file_id, file_name, file_size, file_type, owner_id, input=None):
        cypher = f"MATCH (n:{NodeLabelStr.USER}) WHERE n.user_id = '{owner_id}' MERGE (n1:{NodeLabelStr.PDF_FILE} {{file_id: '{file_id}', file_name: '{file_name}', file_size: {file_size}, file_type: '{file_type}', owner_id: '{owner_id}'}})-[:r]-(n)"
        return CypherDriver.execute_transaction_write(cypher, input)

