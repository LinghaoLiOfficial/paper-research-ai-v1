from utils.database.SqlDriver import SqlDriver


class DataAnalysisMapper:

    @classmethod
    def delete_classification_training_history(cls, params):
        sql = "DELETE FROM my_blog_mysql.da_classification_training_history WHERE task_id = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def delete_best_classification_training_history(cls, params):
        sql = "DELETE FROM my_blog_mysql.da_best_classification_training_history WHERE task_id = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def delete_regression_training_history(cls, params):
        sql = "DELETE FROM my_blog_mysql.da_regression_training_history WHERE task_id = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def delete_best_regression_training_history(cls, params):
        sql = "DELETE FROM my_blog_mysql.da_best_regression_training_history WHERE task_id = %s"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def select_user(cls, params):
        sql = "SELECT * FROM my_blog_mysql.user WHERE user_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_best_classification_training_history_where_task_id_and_study_id_and_run_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.da_best_classification_training_history WHERE task_id = %s AND study_id = %s AND run_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_best_classification_training_history_where_user_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.da_best_classification_training_history WHERE history_owner = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_classification_training_history(cls, params):
        sql = "SELECT * FROM my_blog_mysql.da_classification_training_history WHERE task_id = %s AND study_id = %s AND run_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_best_regression_training_history_where_task_id_and_study_id_and_run_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.da_best_regression_training_history WHERE task_id = %s AND study_id = %s AND run_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_best_regression_training_history_where_user_id(cls, params):
        sql = "SELECT * FROM my_blog_mysql.da_best_regression_training_history WHERE history_owner = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def select_regression_training_history(cls, params):
        sql = "SELECT * FROM my_blog_mysql.da_regression_training_history WHERE task_id = %s AND study_id = %s AND run_id = %s"
        return SqlDriver.execute_read(sql, params)

    @classmethod
    def insert_best_classification_training_history(cls, params):
        sql = "INSERT INTO my_blog_mysql.da_best_classification_training_history (history_id, history_owner, task_id, task_name, study_id, run_id, history_epoch, test_loss, test_accuracy, test_recall, test_precision, test_f1, test_other, history_timestamp, history_params) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s)"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def insert_classification_training_history(cls, params):
        sql = "INSERT INTO my_blog_mysql.da_classification_training_history (history_id, history_owner, task_id, task_name, study_id, run_id, history_epoch, train_loss, train_accuracy, train_recall, train_precision, train_f1, train_other, test_loss, test_accuracy, test_recall, test_precision, test_f1, test_other, history_timestamp, history_params) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s)"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def insert_best_regression_training_history(cls, params):
        sql = "INSERT INTO my_blog_mysql.da_best_regression_training_history (history_id, history_owner, task_id, task_name, study_id, run_id, history_epoch, test_loss, test_mse, test_rmse, test_mae, test_r2, test_other, history_timestamp, history_params) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s)"
        return SqlDriver.execute_write(sql, params)

    @classmethod
    def insert_regression_training_history(cls, params):
        sql = "INSERT INTO my_blog_mysql.da_regression_training_history (history_id, history_owner, task_id, task_name, study_id, run_id, history_epoch, train_loss, train_mse, train_rmse, train_mae, train_r2, train_other, test_loss, test_mse, test_rmse, test_mae, test_r2, test_other, history_timestamp, history_params) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s)"
        return SqlDriver.execute_write(sql, params)

