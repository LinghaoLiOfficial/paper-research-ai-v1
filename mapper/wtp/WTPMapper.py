from utils.database.SqlDriver import SqlDriver


class WTPMapper:

    @classmethod
    def select_training_history(cls):
        sql = f"SELECT * FROM training_history"

        return SqlDriver.execute_read(sql, TrainingHistory)

    @classmethod
    def select_best_training_history(cls):
        sql = f"SELECT * FROM best_training_history"

        return SqlDriver.execute_read(sql, BestTrainingHistory)

    @classmethod
    def select_best_training_history_where_history_owner_and_history_name(cls, history_owner, history_name):
        sql = f"SELECT * FROM best_training_history WHERE history_owner='{history_owner}' AND history_name='{history_name}'"

        return SqlDriver.execute_read(sql, BestTrainingHistory)

    @classmethod
    def insert_training_history(cls,
                                history_id,
                                history_owner,
                                history_name,
                                history_epoch,
                                train_loss,
                                train_accuracy,
                                train_recall,
                                train_precision,
                                train_f1,
                                train_balanced_accuracy_with_recall,
                                train_other,
                                test_loss,
                                test_accuracy,
                                test_recall,
                                test_precision,
                                test_f1,
                                test_balanced_accuracy_with_recall,
                                test_other,
                                history_params,
                                input=None):

        sql = f"INSERT INTO training_history " \
              f"(history_id, history_owner, history_name, history_epoch, train_loss, train_accuracy, train_recall, train_precision, train_f1, train_balanced_accuracy_with_recall, train_other, test_loss, test_accuracy, test_recall, test_precision, test_f1, test_balanced_accuracy_with_recall, test_other, history_timestamp, history_params) VALUES " \
              f"('{history_id}', '{history_owner}', '{history_name}', '{history_epoch}', '{train_loss}', '{train_accuracy}', '{train_recall}', '{train_precision}', '{train_f1}', '{train_balanced_accuracy_with_recall}', '{train_other}', '{test_loss}', '{test_accuracy}', '{test_recall}', '{test_precision}', '{test_f1}', '{test_balanced_accuracy_with_recall}', '{test_other}', NOW(), '{history_params}')"

        return SqlDriver.execute_transaction_write(sql, input)

    @classmethod
    def insert_best_training_history(cls,
                                history_id,
                                history_owner,
                                history_name,
                                history_epoch,
                                test_loss,
                                test_accuracy,
                                test_recall,
                                test_precision,
                                test_f1,
                                test_balanced_accuracy_with_recall,
                                test_other,
                                history_params,
                                input=None):

        sql = f"INSERT INTO best_training_history " \
              f"(history_id, history_owner, history_name, history_epoch, test_loss, test_accuracy, test_recall, test_precision, test_f1, test_balanced_accuracy_with_recall, test_other, history_timestamp, history_params) VALUES " \
              f"('{history_id}', '{history_owner}', '{history_name}', '{history_epoch}', '{test_loss}', '{test_accuracy}', '{test_recall}', '{test_precision}', '{test_f1}', '{test_balanced_accuracy_with_recall}', '{test_other}', NOW(), '{history_params}')"

        return SqlDriver.execute_transaction_write(sql, input)

    @classmethod
    def update_best_training_history(cls,
                                     history_owner,
                                     history_name,
                                     history_epoch,
                                     test_loss,
                                     test_accuracy,
                                     test_recall,
                                     test_precision,
                                     test_f1,
                                     test_balanced_accuracy_with_recall,
                                     test_other,
                                     history_params,
                                     input=None):
        sql = f"UPDATE best_training_history SET history_epoch='{history_epoch}', test_loss='{test_loss}', test_accuracy='{test_accuracy}', test_recall='{test_recall}', test_precision='{test_precision}', test_f1='{test_f1}', test_balanced_accuracy_with_recall='{test_balanced_accuracy_with_recall}', test_other='{test_other}', history_timestamp=NOW(), history_params='{history_params}' WHERE history_owner='{history_owner}' AND history_name='{history_name}'"

        return SqlDriver.execute_transaction_write(sql, input)
