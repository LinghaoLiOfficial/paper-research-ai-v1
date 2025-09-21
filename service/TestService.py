from utils.database.CypherDriver import CypherDriver
from utils.database.SqlDriver import SqlDriver
from entity.common.Resp import Resp


class TestService:

    @classmethod
    def connect_mysql(cls):
        sql_driver_result = SqlDriver.test_connection()
        if not sql_driver_result.check:
            return Resp.build_error()

        return Resp.build_success()

    @classmethod
    def connect_neo4j(cls):
        cypher_driver_result = CypherDriver.test_connection()
        if not cypher_driver_result.check:
            return Resp.build_error()

        return Resp.build_success()

