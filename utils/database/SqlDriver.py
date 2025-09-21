import os
from typing import Type, Optional, Tuple, Any

import mysql.connector

from entity.common.StrucResult import StrucResult


class SqlDriver:

    @classmethod
    def _get_connect(cls):
        return mysql.connector.connect(
            host=os.getenv("MYSQL_CONNECTOR_HOST"),
            user=os.getenv("MYSQL_CONNECTOR_USER"),
            password=os.getenv("MYSQL_CONNECTOR_PASSWORD"),
            database=os.getenv("MYSQL_CONNECTOR_DATABASE"),
            charset="utf8mb4"  # 确保支持特殊字符
        )

    # 测试连接
    @classmethod
    def test_connection(cls):
        connection = cls._get_connect()

        if not connection.is_connected():
            connection.close()
            return StrucResult.build_error()

        connection.close()
        return StrucResult.build_success()

    # 读操作：支持参数化查询
    @classmethod
    def execute_read(
        cls,
        sql: str,
        params  # 新增 params 参数
    ):

        # 转换传入参数格式
        params = tuple(v for k, v in params.items())

        connection = cls._get_connect()
        cursor = connection.cursor()

        try:
            # 使用参数化查询
            cursor.execute(sql, params)  # 关键修改
            results = cursor.fetchall()

            # 获取列名
            columns = [desc[0] for desc in cursor.description]

            # 将结果转换为字典列表
            result_mappings = [dict(zip(columns, row)) for row in results]

        except Exception as e:
            print("Read Error:", e)
            print("Failed SQL:", cursor.statement)  # 打印实际执行的SQL
            cursor.close()
            connection.close()
            return StrucResult.build_error()

        cursor.close()
        connection.close()

        return StrucResult.build_success_with_results(result_mappings)

    # 写操作：支持参数化查询
    @classmethod
    def execute_write(
        cls,
        sql: str,
        params  # 新增 params 参数
    ):

        # 转换传入参数格式
        params = tuple(v for k, v in params.items())

        connection = cls._get_connect()
        cursor = connection.cursor()

        try:
            # 使用参数化查询
            cursor.execute(sql, params)  # 关键修改
            connection.commit()

        except Exception as e:
            print("Write Error:", e)
            print("Failed SQL:", cursor.statement)  # 打印实际执行的SQL
            cursor.close()
            connection.close()
            return StrucResult.build_error()

        cursor.close()
        connection.close()
        return StrucResult.build_success()

    # 事务操作：支持参数化查询
    @classmethod
    def execute_transaction_write(
        cls,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,  # 新增 params 参数
        input: Optional[Tuple[Any, ...]] = None
    ):
        if input is None:
            connection = cls._get_connect()
            cursor = connection.cursor()

            try:
                connection.start_transaction()
                # 使用参数化查询
                cursor.execute(sql, params)  # 关键修改
            except Exception as e:
                print("Transaction Error:", e)
                print("Failed SQL:", cursor.statement)
                cursor.close()
                connection.close()
                return StrucResult.build_error()
        else:
            cursor, connection = input
            try:
                # 使用参数化查询
                cursor.execute(sql, params)  # 关键修改
            except Exception as e:
                print("Transaction Error:", e)
                print("Failed SQL:", cursor.statement)
                cursor.close()
                connection.close()
                return StrucResult.build_error()

        return StrucResult.build_success_with_results((cursor, connection))

    @classmethod
    def rollback(cls, input):
        cursor, connection = input

        try:
            connection.rollback()

        except Exception as e:
            print(e)
            cursor.close()
            connection.close()
            return StrucResult.build_error()

        cursor.close()
        connection.close()
        return StrucResult.build_success()

    @classmethod
    def close(cls, input):
        cursor, connection = input

        try:
            connection.commit()

        except Exception as e:
            print(e)
            cursor.close()
            connection.close()
            return StrucResult.build_error()

        cursor.close()
        connection.close()
        return StrucResult.build_success()
