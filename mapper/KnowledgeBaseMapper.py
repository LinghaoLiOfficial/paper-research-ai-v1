from utils.database.CypherDriver import CypherDriver
from utils.database.SqlDriver import SqlDriver


class KnowledgeBaseMapper:

    @classmethod
    def select_file_where_file_name(cls, params):
        sql = "SELECT * FROM my_blog_mysql.file WHERE file_name = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_file_where_belong_folder(cls, params):
        sql = "SELECT * FROM my_blog_mysql.file WHERE belong_folder = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_file_where_file_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.file WHERE file_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_file_where_owner_id_and_file_type(cls, params):
        sql = "SELECT * FROM my_blog_mysql.file WHERE owner_id = %s AND file_type = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_file_where_owner_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.file WHERE owner_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_auth_where_index(cls, params):
        sql = "SELECT * FROM my_blog_mysql.auth WHERE auth_index = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_user(cls, params):
        sql = "SELECT * FROM my_blog_mysql.user WHERE user_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def delete_file(cls, params):
        sql = "DELETE FROM my_blog_mysql.file WHERE file_id = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def select_file_where_user_id_order_by_time_desc(cls, params):
        sql = "SELECT * from my_blog_mysql.file WHERE owner_id = %s AND belong_folder = %s ORDER BY create_timestamp DESC"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def insert_file(cls, params):
        sql = "INSERT INTO my_blog_mysql.file (file_id, file_name, file_size, file_type, owner_id, create_timestamp, belong_folder, file_label) VALUES (%s, %s, %s, %s, %s, %s, %s, '')"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def delete_all_file_node_with_node_children(cls, params):
        cypher = "MATCH (n:file {id: $file_id})-[*0..]->(n1) DETACH DELETE n1 "
        return CypherDriver.execute_write(cypher, params)

    @classmethod
    def delete_all_file_node_children(cls, params):
        cypher = "MATCH (n:file {id: $file_id})-[*1..]->(n1) DETACH DELETE n1 "
        return CypherDriver.execute_write(cypher, params)

    @classmethod
    def match_file_node_on_file_id(cls, params):
        cypher = """
            MATCH (root:file)-[*0..1]->(child)
            WHERE root.id = $file_id
            RETURN
                root.id AS id,
                root.file_name AS label,
                collect(child.id) AS children_ids
        """
        return CypherDriver.execute_read(cypher, params)

    @classmethod
    def match_all_image_node_on_file_id(cls, params):
        cypher = """
            MATCH (root:file)-[*1..]->(node:image)
            WHERE root.id = $file_id
            MATCH (node)-[*0..1]->(child)
            RETURN
                node.id AS id,
                node.image_name AS image_name,
                node.image_width AS image_width,
                node.image_height AS image_height,
                collect(child.id) AS children_ids
        """
        return CypherDriver.execute_read(cypher, params)

    @classmethod
    def match_all_text_node_on_file_id(cls, params):
        cypher = """
            MATCH (root:file)-[*1..]->(node:text)
            WHERE root.id = $file_id
            MATCH (node)-[*0..1]->(child)
            RETURN
                node.id AS id,
                node.text_name AS label,
                collect(child.id) AS children_ids
        """
        return CypherDriver.execute_read(cypher, params)

    @classmethod
    def match_file_node_where_file_type(cls, params):
        cypher = f"MATCH (n:user)-[]-(n1:file) WHERE n.id = $user_id and n1.file_type = $file_type RETURN DISTINCT n1 AS properties"
        return CypherDriver.execute_read(cypher, params)

    @classmethod
    def merge_file_node(cls, params):
        cypher = "MATCH (n:user) WHERE n.id = $user_id MERGE (n1:file {id: $file_id, file_name: $file_name, file_type: $file_type})<-[:r]-(n)"
        return CypherDriver.execute_write(cypher, params)

    @classmethod
    def merge_text_node(cls, params):
        cypher = "MATCH (n) WHERE n.id = $parent_id MERGE (n1:text {id: $child_text_id, text_name: $child_text_name})<-[:r]-(n)"
        return CypherDriver.execute_write(cypher, params)

    @classmethod
    def merge_image_node(cls, params):
        cypher = "MATCH (n) WHERE n.id = $parent_id MERGE (n1:image {id: $child_image_id, image_name: $child_image_name, image_width: $child_image_width, image_height: $child_image_height})<-[:r]-(n)"
        return CypherDriver.execute_write(cypher, params)
