from utils.database.SqlDriver import SqlDriver


class WebCrawlMapper:

    @classmethod
    def select_progress_where_id(cls, id):
        sql = f"SELECT * FROM my_blog_mysql.web_crawl_progress WHERE progress_id = '{id}'"
        return SqlDriver.execute_read(sql)

    @classmethod
    def select_progress_where_owner(cls, owner):
        sql = f"SELECT * FROM my_blog_mysql.web_crawl_progress WHERE progress_owner = '{owner}'"
        return SqlDriver.execute_read(sql)

    @classmethod
    def insert_progress(cls, id, owner, name=''):
        sql = f"INSERT INTO my_blog_mysql.web_crawl_progress (progress_id, progress_owner, progress_name, progress_step_1, progress_step_2, progress_step_3, progress_step_4) VALUES ('{id}', '{owner}', '{name}', 0, 0, 0, 0)"
        return SqlDriver.execute_write(sql)

    @classmethod
    def update_progress_set_step(cls, step_name, step_value, id):
        sql = f"UPDATE my_blog_mysql.web_crawl_progress SET {step_name} = {step_value} WHERE progress_id = '{id}'"
        return SqlDriver.execute_write(sql)
