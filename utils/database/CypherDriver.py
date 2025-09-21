from neo4j import GraphDatabase
import os
import subprocess

from entity.common.StrucResult import StrucResult


class CypherDriver:

    driver = None

    @classmethod
    def _get_connect(cls):
        if cls.driver is None:

            cls.driver = GraphDatabase.driver(
                uri=os.getenv("NEO4J_CONNECTOR_URI"),
                auth=(
                    os.getenv("NEO4J_CONNECTOR_AUTH_USER"),
                    os.getenv("NEO4J_CONNECTOR_AUTH_PASSWORD")
                )
            )

        return cls.driver

    # 测试连接
    @classmethod
    def test_connection(cls):
        try:
            cls._get_connect().verify_connectivity()

        except Exception as e:
            print(e)
            return StrucResult.build_error()

        return StrucResult.build_success()

    @classmethod
    def _execute_write_cypher(cls, graph, cypher, params=None):
        # 添加 parameters 参数
        graph.run(cypher, parameters=params)

    @classmethod
    def _execute_read_cypher(cls, graph, cypher, params=None):
        # 添加 parameters 参数
        result = graph.run(cypher, parameters=params)
        values_list = [{k: v for k, v in record.items()} for record in result]
        return values_list

    @classmethod
    def execute_read(cls, cypher: str, params=None):  # 增加 params 参数
        session = cls._get_connect().session()
        try:
            # 传递 params 到 _execute_read_cypher
            results = session.execute_read(cls._execute_read_cypher, cypher, params)
        except Exception as e:
            print(e)
            session.close()
            return StrucResult.build_error()
        session.close()
        return StrucResult.build_success_with_results(results)

    @classmethod
    def execute_write(cls, cypher: str, params=None):  # 增加 params 参数
        session = cls._get_connect().session()
        try:
            # 传递 params 到 _execute_write_cypher
            session.execute_write(cls._execute_write_cypher, cypher, params)
        except Exception as e:
            print(e)
            session.close()
            return StrucResult.build_error()
        session.close()
        return StrucResult.build_success()

    @classmethod
    def execute_transaction_write(cls, cypher: str, params=None, input=None):
        if input is None:
            session = cls._get_connect().session()
            transaction = session.begin_transaction()
            try:
                # 传递 parameters 参数
                transaction.run(cypher, parameters=params)
            except Exception as e:
                print(e)
                transaction.close()
                session.close()
                return StrucResult.build_error()
        else:
            transaction, session = input
            try:
                # 传递 parameters 参数
                transaction.run(cypher, parameters=params)
            except Exception as e:
                print(e)
                transaction.close()
                session.close()
                return StrucResult.build_error()
        return StrucResult.build_success_with_results((transaction, session))

    @classmethod
    def rollback(cls, input):
        transaction, session = input

        try:
            transaction.rollback()

        except Exception as e:
            print(e)
            transaction.close()
            session.close()
            return StrucResult.build_error()

        transaction.close()
        session.close()
        return StrucResult.build_success()

    @classmethod
    def close(cls, input):
        transaction, session = input

        try:
            transaction.commit()

        except Exception as e:
            print(e)
            transaction.close()
            session.close()
            return StrucResult.build_error()

        transaction.close()
        session.close()
        return StrucResult.build_success()

