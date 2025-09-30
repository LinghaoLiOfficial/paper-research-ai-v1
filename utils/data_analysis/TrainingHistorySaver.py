import copy
import json
from datetime import datetime

from mapper.wtp.WTPMapper import WTPMapper
from utils.common.RandomStrGenerator import RandomStrGenerator
from utils.common.TimeParser import TimeParser
from utils.database.SqlDriver import SqlDriver
from entity.common.Resp import Resp


class TrainingHistorySaver:

    @classmethod
    def convert(cls, training_history_result, best_training_history_result):

        if len(training_history_result) == 0:
            return {
                "head": [],
                "body": []
            }

        train_head = [x for x in training_history_result[0].keys() if x not in ["history_id", "history_owner", "history_params"]]
        best_head = [x for x in best_training_history_result[0].keys() if x not in ["history_id", "history_owner", "history_params"]]

        history_params = training_history_result[0]['history_params']

        body = []
        for best_item in best_training_history_result:

            best_item['history_timestamp'] = TimeParser.convert_time_format(best_item['history_timestamp'])
            best_item['history_epoch'] = str(best_item['history_epoch'])

            current_item = [best_item[k] for k in best_item.keys() if k not in ["history_id", "history_owner", "history_params"]]

            current_history_owner = best_item['history_owner']
            current_history_name = best_item['task_name']

            children = copy.deepcopy([train_item for train_item in training_history_result if train_item['history_owner'] == current_history_owner and train_item['task_name'] == current_history_name])

            children = sorted(children, key=lambda x: int(x['history_epoch']), reverse=True)

            for item in children:
                item['history_epoch'] = str(item['history_epoch'])
                item['history_timestamp'] = TimeParser.convert_time_format(item['history_timestamp'])

            current_item.append([[x[k] for k in x.keys() if k not in ["history_id", "history_owner", "history_params"]] for x in children])

            body.append(current_item)

        body = sorted(body, key=lambda v: datetime.strptime(v[10], '%Y-%m-%d %H:%M:%S'), reverse=True)

        return {
            "trainHead": train_head,
            "bestHead": best_head,
            "body": body,
            "history_params": history_params
        }

    @classmethod
    def load(cls):
        mysql_result = WTPMapper.select_training_history()
        if not mysql_result.check:
            return Resp.build_db_error()

        mysql_result1 = WTPMapper.select_best_training_history()
        if not mysql_result1.check:
            return Resp.build_db_error()

        training_history = cls.convert(
            training_history_result=mysql_result.get_data_on_results(),
            best_training_history_result=mysql_result1.get_data_on_results()
        )

        return training_history

    @classmethod
    def auto_save_into_database(cls,
                                history_owner,
                                history_name,
                                history_epoch,
                                best_indicators_result,
                                current_config):

        DEFAULT_VALUE = "-"

        history_epoch = f"EPOCH {history_epoch + 1}"

        history_id = RandomStrGenerator.generate_uuid()

        mysql_result1 = WTPMapper.select_best_training_history_where_history_owner_and_history_name(
            history_owner=history_owner,
            history_name=history_name
        )
        if not mysql_result1.check:
            return Resp.build_db_error()

        has_best_training_history_status = mysql_result1.get_data_on_results()
        if has_best_training_history_status:

            mysql_result2 = WTPMapper.update_best_training_history(
                history_owner=history_owner,
                history_name=history_name,
                history_epoch=best_indicators_result["epoch"] + 1,
                test_loss=best_indicators_result["loss"],
                test_accuracy=best_indicators_result["accuracy"],
                test_recall=best_indicators_result["recall"],
                test_precision=best_indicators_result["precision"],
                test_f1=best_indicators_result["f1"],
                test_balanced_accuracy_with_recall=best_indicators_result["balanced_accuracy_with_recall"],
                test_other=DEFAULT_VALUE,
                history_params=json.dumps(current_config)
            )
            if not mysql_result2.check:
                return Resp.build_db_error()

        else:

            best_history_id = RandomStrGenerator.generate_uuid()

            mysql_result2 = WTPMapper.insert_best_training_history(
                history_id=best_history_id,
                history_owner=history_owner,
                history_name=history_name,
                history_epoch=best_indicators_result["epoch"] + 1,
                test_loss=best_indicators_result["loss"],
                test_accuracy=best_indicators_result["accuracy"],
                test_recall=best_indicators_result["recall"],
                test_precision=best_indicators_result["precision"],
                test_f1=best_indicators_result["f1"],
                test_balanced_accuracy_with_recall=best_indicators_result["balanced_accuracy_with_recall"],
                test_other=DEFAULT_VALUE,
                history_params=json.dumps(current_config),
            )
            if not mysql_result2.check:
                return Resp.build_db_error()

        sql_driver_result2 = SqlDriver.close(mysql_result2.get_data_on_results())

        return Resp.build_success()

    @classmethod
    def save_into_database(cls,
                           history_owner,
                           history_name,
                           history_epoch,
                           train_indicators_result,
                           test_indicators_result,
                           best_indicators_result,
                           current_config):

        DEFAULT_VALUE = "-"

        history_epoch = f"EPOCH {history_epoch + 1}"

        history_id = RandomStrGenerator.generate_uuid()

        mysql_result = WTPMapper.insert_training_history(
            history_id=history_id,
            history_owner=history_owner,
            history_name=history_name,
            history_epoch=history_epoch,
            train_loss=train_indicators_result["loss"],
            train_accuracy=train_indicators_result["accuracy"],
            train_recall=train_indicators_result["recall"],
            train_precision=train_indicators_result["precision"],
            train_f1=train_indicators_result["f1"],
            train_balanced_accuracy_with_recall=train_indicators_result["balanced_accuracy_with_recall"],
            train_other=train_indicators_result["other"] if "other" in train_indicators_result.keys() else DEFAULT_VALUE,
            test_loss=test_indicators_result["loss"],
            test_accuracy=test_indicators_result["accuracy"],
            test_recall=test_indicators_result["recall"],
            test_precision=test_indicators_result["precision"],
            test_f1=test_indicators_result["f1"],
            test_balanced_accuracy_with_recall=test_indicators_result["balanced_accuracy_with_recall"],
            test_other=DEFAULT_VALUE,
            history_params=json.dumps(current_config)
        )
        if not mysql_result.check:
            return Resp.build_db_error()

        mysql_result1 = WTPMapper.select_best_training_history_where_history_owner_and_history_name(
            history_owner=history_owner,
            history_name=history_name
        )
        if not mysql_result1.check:
            return Resp.build_db_error()

        has_best_training_history_status = mysql_result1.get_data_on_results()
        if has_best_training_history_status:

            mysql_result2 = WTPMapper.update_best_training_history(
                history_owner=history_owner,
                history_name=history_name,
                history_epoch=best_indicators_result["epoch"] + 1,
                test_loss=best_indicators_result["loss"],
                test_accuracy=best_indicators_result["accuracy"],
                test_recall=best_indicators_result["recall"],
                test_precision=best_indicators_result["precision"],
                test_f1=best_indicators_result["f1"],
                test_balanced_accuracy_with_recall=best_indicators_result["balanced_accuracy_with_recall"],
                test_other=DEFAULT_VALUE,
                history_params=json.dumps(current_config),
                input=mysql_result.get_data_on_results()
            )
            if not mysql_result2.check:
                sql_driver_result = SqlDriver.rollback(mysql_result.get_data_on_results())
                return Resp.build_db_error()

        else:

            best_history_id = RandomStrGenerator.generate_uuid()

            mysql_result2 = WTPMapper.insert_best_training_history(
                history_id=best_history_id,
                history_owner=history_owner,
                history_name=history_name,
                history_epoch=best_indicators_result["epoch"] + 1,
                test_loss=best_indicators_result["loss"],
                test_accuracy=best_indicators_result["accuracy"],
                test_recall=best_indicators_result["recall"],
                test_precision=best_indicators_result["precision"],
                test_f1=best_indicators_result["f1"],
                test_balanced_accuracy_with_recall=best_indicators_result["balanced_accuracy_with_recall"],
                test_other=DEFAULT_VALUE,
                history_params=json.dumps(current_config),
                input=mysql_result.get_data_on_results()
            )
            if not mysql_result2.check:
                sql_driver_result = SqlDriver.rollback(mysql_result.get_data_on_results())
                return Resp.build_db_error()

        sql_driver_result = SqlDriver.close(mysql_result.get_data_on_results())
        sql_driver_result2 = SqlDriver.close(mysql_result2.get_data_on_results())

        return Resp.build_success()
