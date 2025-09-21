from utils.database.CypherDriver import CypherDriver
from utils.database.SqlDriver import SqlDriver


class ScienceResearchMapper:

    @classmethod
    def select_file_where_file_name(cls, params):
        sql = "SELECT * FROM my_blog_mysql.file WHERE file_name = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def delete_theme_higher_explaining(cls, params):
        sql = "DELETE FROM my_blog_mysql.sr_theme_higher_explaining WHERE task_id = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def delete_theme_explaining(cls, params):
        sql = "DELETE FROM my_blog_mysql.sr_theme_explaining WHERE task_id = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def update_theme_higher_explaining(cls, params):
        sql = "UPDATE my_blog_mysql.sr_theme_higher_explaining SET research_zh_questions = %s, research_zh_methods = %s, research_zh_contributions = %s, research_zh_limitations = %s WHERE task_id = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def update_theme_explaining(cls, params):
        sql = "UPDATE my_blog_mysql.sr_theme_explaining SET research_zh_questions = %s, research_zh_methods = %s, research_zh_contributions = %s, research_zh_limitations = %s WHERE task_id = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def select_theme_higher_explaining(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_theme_higher_explaining WHERE task_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_theme_explaining(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_theme_explaining WHERE task_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def insert_theme_explaining(cls, params):
        sql = "INSERT INTO my_blog_mysql.sr_theme_explaining (explaining_id, user_id, task_id, research_questions, research_methods, research_contributions, research_limitations) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def insert_theme_higher_explaining(cls, params):
        sql = "INSERT INTO my_blog_mysql.sr_theme_higher_explaining (explaining_id, user_id, task_id, research_questions, research_methods, research_contributions, research_limitations) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def select_user_where_user_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.user WHERE user_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def on_task_id_get_user_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_paper_search WHERE task_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def delete_graph(cls, params):
        sql = "DELETE FROM my_blog_mysql.sr_graph WHERE task_id = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def match_higher_keywords(cls, params, content_perspective):
        cypher = f"""
            MATCH (n:science_theme {{id: $task_id}})-[r:{content_perspective}]->(n1:science_theme) 
            RETURN n1.name AS name
        """
        return CypherDriver.execute_read(cypher, params)

    @classmethod
    def match_concrete_level_concrete_year_science_theme_node(cls, params, content_perspective):
        cypher = f"""
            MATCH (n:science_theme {{id: $task_id}})-[r:{content_perspective}]->(n1:science_theme {{name: $higher_theme_name}})-[r1:{content_perspective}]->(n2:science_theme)-[:{content_perspective}]->(n3:science_paper) 
            RETURN 
                n2.name AS theme, 
                collect(n3.name) AS papers, 
                collect(r.weight) AS parent_weight, 
                collect(r1.weight) AS weight 
        """

        return CypherDriver.execute_read(cypher, params)

    @classmethod
    def select_graph(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_graph WHERE task_id = %s and content_perspective = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def insert_graph(cls, params):
        sql = "INSERT INTO my_blog_mysql.sr_graph (report_id, task_id, content_perspective, paragraph) VALUES (%s, %s, %s, %s)"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def delete_old_bridge_theme(cls, params, edge_type):
        cypher = f"""
            MATCH (n:user {{id: $user_id}})-[]->(n1:science_theme {{id: $task_id}})-[:{edge_type}]->(n2:science_theme {{id: $higher_theme_name}})-[r:{edge_type}]->(n3:science_theme {{id: $theme_name}})-[r1:{edge_type}]->(n4:science_paper)
            DETACH DELETE n3
        """
        return CypherDriver.execute_write(cypher, params)

    @classmethod
    def new_bridge_to_delete_old_data(cls, params, edge_type):
        cypher = f"""
            MATCH (n:user {{id: $user_id}})-[]->(n1:science_theme {{id: $task_id}})-[r0:{edge_type}]->(n2:science_theme {{name: $higher_theme_name}})-[r:{edge_type}]->(n3:science_theme {{name: $theme_name}})-[r1:{edge_type}]->(n4:science_paper)  
            DETACH DELETE n3
        """
        return CypherDriver.execute_write(cypher, params)

    @classmethod
    def new_bridge_to_create_new_data(cls, params, edge_type):
        cypher = f"""
            MATCH (n:user {{id: $user_id}})-[]->(n1:science_theme {{id: $task_id}})-[r0:{edge_type}]->(n2:science_theme {{name: $higher_theme_name}}) 
            MATCH (n4:science_paper {{id: $paper_id}})
            MERGE (n5:science_theme {{name: $bridge_name}}) 
            ON CREATE SET n5.id = $bridge_id, n5.theme_explaining = $theme_explaining 
            MERGE (n2)-[:{edge_type} {{weight: $left_weight}}]->(n5)-[:{edge_type} {{weight: $right_weight}}]->(n4) 
        """
        return CypherDriver.execute_write(cypher, params)

    @classmethod
    def new_bridge_to_get_old_data(cls, params, edge_type):
        cypher = f"""
            MATCH (n:user {{id: $user_id}})-[]->(n1:science_theme {{id: $task_id}})-[r0:{edge_type}]->(n2:science_theme {{name: $higher_theme_name}})-[r:{edge_type}]->(n3:science_theme {{name: $theme_name}})-[r1:{edge_type}]->(n4:science_paper)  
            RETURN r.weight AS left_weight, r1.weight AS right_weight, n4.id AS paper_id 
        """

        return CypherDriver.execute_read(cypher, params)

    @classmethod
    def delete_theme_filtered(cls, params):
        sql = "DELETE FROM my_blog_mysql.sr_theme_filtered WHERE task_id = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def select_theme_filtered_where_user_id_and_task_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_theme_filtered WHERE user_id = %s AND task_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def insert_theme_filtered(cls, params):
        sql = "INSERT INTO my_blog_mysql.sr_theme_filtered (filtered_id, user_id, task_id, create_timestamp, research_questions, research_methods, research_contributions, research_limitations) VALUES (%s, %s, %s, NOW(), %s, %s, %s, %s)"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def update_paper_search_set_task_status(cls, params):
        sql = "UPDATE my_blog_mysql.sr_paper_search SET task_status = %s WHERE search_id = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def update_paper_search_set_paper_id(cls, params):
        sql = "UPDATE my_blog_mysql.sr_paper_search SET paper_id = %s WHERE search_id = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def delete_science_paper_and_science_theme_node(cls, params):
        cypher = "MATCH (n:science_theme {id: $task_id})-[*0..]->(n1:science_paper | science_theme) DETACH DELETE n1"
        return CypherDriver.execute_write(cypher, params)

    @classmethod
    def delete_paper_search(cls, params):
        sql = "DELETE FROM my_blog_mysql.sr_paper_search WHERE task_id = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def delete_theme_characterize(cls, params):
        sql = "DELETE FROM my_blog_mysql.sr_theme_characterize WHERE task_id = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def delete_theme_translate(cls, params):
        sql = "DELETE FROM my_blog_mysql.sr_theme_translate WHERE task_id = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def select_paper_search_where_user_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_paper_search WHERE user_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_paper(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_paper"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_theme_translate(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_theme_translate"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def insert_theme_translate(cls, params):
        sql = "INSERT INTO my_blog_mysql.sr_theme_translate (theme_id, theme_name, theme_zh_name, user_id, task_id) VALUES (%s, %s, %s, %s, %s)"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def select_theme_translate_where_theme_name(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_theme_translate WHERE theme_name = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_theme_characterize_where_user_id_and_task_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_theme_characterize WHERE user_id = %s AND task_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def insert_theme_characterize(cls, params):
        sql = "INSERT INTO my_blog_mysql.sr_theme_characterize (characterize_id, user_id, task_id, create_timestamp, research_questions, research_methods, research_contributions, research_limitations) VALUES (%s, %s, %s, NOW(), %s, %s, %s, %s)"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def update_paper(cls, params):
        sql = """
            UPDATE my_blog_mysql.sr_paper 
            SET research_questions = %s, research_questions_explaining = %s, research_methods = %s, research_methods_explaining = %s, research_contributions = %s, research_contributions_explaining = %s, research_limitations = %s, research_limitations_explaining = %s, paper_summary = %s, paper_zh_summary = %s 
            WHERE paper_id = %s
        """
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def insert_paper_search(cls, params):
        sql = "INSERT INTO my_blog_mysql.sr_paper_search (search_id, user_id, task_id, task_name, search_words, paper_num, create_timestamp, task_type, task_status, pdf_path) VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, 'ongoing', %s)"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def match_all_theme_node(cls, params):
        cypher = "MATCH (n:science_theme) RETURN n"
        return CypherDriver.execute_read(cypher, params)

    @classmethod
    def match_paper_node_where_id(cls, params):
        cypher = "MATCH (n:science_paper {id: $paper_id}) RETURN n"
        return CypherDriver.execute_read(cypher, params)

    @classmethod
    def select_paper_where_paper_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_paper WHERE paper_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_paper_summary_where_paper_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_paper_summary WHERE paper_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def match_all_edge_on_root_id(cls, params, edge_type):
        cypher = f"""
            MATCH (root:science_theme {{id: $root_id}})-[:{edge_type}*0..]->(n1)-[r:{edge_type}]->(n2)
            RETURN 
                n1.id AS source,
                n2.id AS target,
                r.weight AS weight
        """
        return CypherDriver.execute_read(cypher, params)

    @classmethod
    def match_all_paper_node_on_root_id(cls, params, edge_type):
        cypher = f"""
            MATCH (root:science_theme)-[:{edge_type}*1..]->(node:science_paper)
            WHERE root.id = $root_id
            MATCH (node)-[*0..1]->(child)
            RETURN
                node.id AS id,
                node.name AS label,
                '论文' AS type,
                collect(child.id) AS children_ids
        """
        return CypherDriver.execute_read(cypher, params)

    @classmethod
    def match_all_theme_node_on_root_id(cls, params, edge_type):
        cypher = f"""
            MATCH (root:science_theme)-[:{edge_type}*0..]->(node:science_theme)
            WHERE root.id = $root_id
            MATCH (node)-[*0..1]->(child)
            RETURN
                node.id AS id,
                node.name AS label,
                '关键词' AS type,
                node.theme_explaining AS theme_explaining,
                collect(child.id) AS children_ids
        """
        return CypherDriver.execute_read(cypher, params)

    @classmethod
    def select_paper_summary_where_user_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_paper_summary WHERE user_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def merge_theme(cls, params, edge_type):
        cypher = f"""
            MATCH (n:user {{id: $user_id}})-[]->(n1:science_theme {{id: $task_id}})-[]->(n2:science_theme {{id: $higher_theme_id}})
            MERGE (n2)-[r:{edge_type} {{weight: $edge_weight}}]->(n3:science_theme {{id: $theme_id, name: $theme_name, theme_explaining: $theme_explaining}})
        """
        return CypherDriver.execute_write(cypher, params)

    @classmethod
    def merge_science_higher_theme(cls, params, edge_type):
        cypher = f"""
            MATCH (n:user {{id: $user_id}})-[]->(n1:science_theme {{id: $task_id}})
            MERGE (n1)-[r:{edge_type} {{weight: $edge_weight}}]->(n2:science_theme {{id: $theme_id, name: $theme_name, theme_explaining: $theme_explaining}})
        """
        return CypherDriver.execute_write(cypher, params)

    @classmethod
    def select_paper_higher_summary_where_user_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_paper_higher_summary WHERE user_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def insert_paper_higher_summary(cls, params):
        sql = "INSERT INTO my_blog_mysql.sr_paper_higher_summary (user_id, research_questions, research_methods, research_contributions, research_limitations) VALUES (%s, %s, %s, %s, %s)"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def select_community(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_community WHERE node_owner = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def insert_community(cls, params):
        sql = "INSERT INTO my_blog_mysql.sr_community (node_id, node_label, node_name, node_owner, node_belong_community) VALUES (%s, %s, %s, %s, %s)"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def match_science_paper_node(cls, params):
        cypher = "MATCH (n:science_paper) RETURN n"
        return CypherDriver.execute_read(cypher, params)

    @classmethod
    def match_all_science_nodes(cls, params):
        cypher = "MATCH (n:science_theme {status: 'search_0'})-[*1..]->(n1)-[r]->(n2) RETURN n1.id AS source, n2.id AS target, COALESCE(r.weight, 1.0) AS weight, n1.name AS source_name, n2.name AS target_name, labels(n1) AS source_labels, labels(n2) AS target_labels"
        return CypherDriver.execute_read(cypher, params)

    @classmethod
    def match_science_theme_node(cls, params):
        cypher = "MATCH (n:science_theme) WHERE n.name = $theme_name RETURN n"
        return CypherDriver.execute_read(cypher, params)

    @classmethod
    def after_merge_science_theme_node_on_theme_node(cls, params):
        cypher = "MATCH (n:science_theme) WHERE n.name = $science_theme_name MATCH (n1:science_theme) WHERE n1.name = $characterized_science_theme_name MERGE (n1)<-[:r {weight: $edge_weight}]-(n)"
        return CypherDriver.execute_write(cypher, params)

    @classmethod
    def first_merge_science_theme_node_on_theme_node(cls, params):
        cypher = "MATCH (n:science_theme) WHERE n.name = $science_theme_name MERGE (n1:science_theme {id: $characterized_science_theme_id, name: $characterized_science_theme_name, status: 'characterize'})<-[:r {weight: $edge_weight}]-(n)"
        return CypherDriver.execute_write(cypher, params)

    @classmethod
    def merge_science_paper_node(cls, params, edge_type):
        cypher = f"""
            MATCH (n:science_theme {{id: $theme_id}})
            MERGE (n)-[r:{edge_type} {{weight: $edge_weight}}]->(n1:science_paper {{name: $paper_name, id: $paper_id}})
        """
        return CypherDriver.execute_write(cypher, params)

    @classmethod
    def match_science_theme_node_on_user_node(cls, params):
        cypher = "MATCH (n:user)-[:r]->(n1:science_theme) WHERE n.id = $user_id AND n1.name = $science_theme_name RETURN n1"
        return CypherDriver.execute_read(cypher, params)

    @classmethod
    def match_concrete_level_science_theme_node(cls, params, level=1):
        cypher = f"MATCH (n:user)-[:r*{int(2 * level + 1)}..]->(n1:science_theme) WHERE n.id = $user_id RETURN n1"
        return CypherDriver.execute_read(cypher, params)

    @classmethod
    def merge_science_theme_node_on_user_node_to_search(cls, params):
        cypher = "MATCH (n:user) WHERE n.id = $user_id MERGE (n1:science_theme {id: $science_theme_id, name: $science_theme_name, status: 'search_0'})<-[:r]-(n)"
        return CypherDriver.execute_write(cypher, params)

    @classmethod
    def merge_science_theme_node_on_paper_node(cls, params):
        cypher = "MATCH (n:science_paper) WHERE n.id = $paper_id MERGE (n1:science_theme {id: $science_theme_id, name: $science_theme_name})<-[:r {weight: $edge_weight}]-(n)"
        return CypherDriver.execute_write(cypher, params)

    @classmethod
    def update_paper_summaries_where_user_id(cls, params):
        sql = "UPDATE my_blog_mysql.sr_paper_summary SET research_themes = %s WHERE paper_id = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def select_paper_summaries_where_user_id(cls, params):
        sql = """
            SELECT my_blog_mysql.sr_paper_summary.* 
            FROM my_blog_mysql.sr_paper 
            JOIN my_blog_mysql.sr_paper_summary ON my_blog_mysql.sr_paper.paper_id = my_blog_mysql.sr_paper_summary.paper_id
            WHERE my_blog_mysql.sr_paper.paper_owner = %s
        """
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_paper_where_user_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_paper"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_paper_search_where_user_id_and_task_id(cls, params):
        sql = """
            SELECT * FROM my_blog_mysql.sr_paper_search a 
            LEFT JOIN my_blog_mysql.sr_paper b 
            ON a.paper_id = b.paper_id 
            WHERE a.user_id = %s AND a.task_id = %s 
        """
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_paper_where_title(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_paper WHERE paper_title = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_paper_summary_where_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_paper_summary WHERE paper_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_paper_where_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.sr_paper WHERE paper_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def insert_paper(cls, params):
        sql = "INSERT INTO my_blog_mysql.sr_paper (paper_id, paper_title, paper_zh_title, paper_authors, paper_publish_time, paper_abstract, paper_zh_abstract, paper_url, paper_doi, paper_journal, paper_pdf_url, paper_language) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def insert_paper_summaries(cls, params):
        sql = "INSERT INTO my_blog_mysql.sr_paper_summary (paper_id, paper_title, user_id, research_questions, research_methods, research_contributions, research_limitations, paper_summary) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        return SqlDriver.execute_write(sql, params)
