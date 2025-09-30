from config.neo4j.NodeLabelStr import NodeLabelStr
from utils.database.CypherDriver import CypherDriver
from utils.database.SqlDriver import SqlDriver


class LoginMapper:

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
    def reset_paper_search(cls, params):
        sql = "UPDATE my_blog_mysql.sr_paper_search SET task_status = 'failed' WHERE task_status = 'ongoing'"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def insert_activation(cls, params):
        sql = "INSERT INTO my_blog_mysql.register_activation (code_id, activation_code) VALUES (%s, %s)"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def select_all_activation(cls, params):
        sql = "SELECT * FROM my_blog_mysql.register_activation"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_user_where_user_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.user WHERE user_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def delete_activation(cls, params):
        sql = "DELETE FROM my_blog_mysql.register_activation WHERE activation_code = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def select_activation(cls, params):
        sql = "SELECT * FROM my_blog_mysql.register_activation WHERE activation_code = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_all_from_user_where_email(cls, params):
        sql = "select * from my_blog_mysql.user where user_email = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_all_from_user_where_username(cls, params):
        sql = "select * from my_blog_mysql.user where username = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_all_from_user_where_username_password(cls, params):
        sql = f"select * from my_blog_mysql.user where username = %s and password = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_all_from_validation_where_email(cls, params):
        sql = "select * from my_blog_mysql.register_validation where validation_email = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def transaction_insert_user(cls, params):
        sql = "insert into my_blog_mysql.user (user_id, username, password, user_auth, user_create_timestamp, user_email, user_salt, user_activation) values (%s, %s, %s, %s, NOW(), %s, %s, %s)"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def transaction_insert_validation(cls, params):
        sql = "insert into my_blog_mysql.register_validation (validation_id, validation_email, validation_code, validation_create_timestamp) values (%s, %s, %s, NOW())"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def transaction_delete_validation(cls, params):
        sql = "delete from my_blog_mysql.register_validation where validation_email = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def transaction_merge_user_node(cls, params):
        cypher = "MERGE (n:user {id: $user_id, username: $username})"
        return CypherDriver.execute_write(cypher, params)


