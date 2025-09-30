from config.neo4j.NodeIdStr import NodeIdStr
from config.neo4j.NodeNameStr import NodeNameStr
from mapper.PersonalStudyBoostingMapper.PSBMapper import PSBMapper
from utils.common.FileSaver import FileSaver
from utils.common.FormatConverter import FormatConverter
from utils.common.GraphParser import GraphParser
from utils.common.RandomStrGenerator import RandomStrGenerator
from utils.common.TextTranslator import TextTranslator
from utils.database.CypherDriver import CypherDriver
from utils.database.SqlDriver import SqlDriver
from utils.common.JWTParser import JWTParser
from entity.common.Resp import Resp


class PSBService:

    @classmethod
    def delete_node(cls, token, node_id):
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        neo4j_result = PSBMapper.delete_node_from_id(
            node_id=node_id
        )
        if not neo4j_result.check:
            return Resp.build_db_error()

        return Resp.build_success()

    @classmethod
    def update_node_property(cls, token, node_id, node_key, node_value):
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        neo4j_result = PSBMapper.set_node_property_value_from_key(
            node_id=node_id,
            node_key=node_key,
            node_value=node_value
        )
        if not neo4j_result.check:
            return Resp.build_db_error()

        return Resp.build_success()

    @classmethod
    def search_node(cls, token, text_to_search):
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        neo4j_result = PSBMapper.match_all_from_all_name(
            user_id=user_id,
            text_to_search=text_to_search
        )
        if not neo4j_result.check:
            return Resp.build_db_error()

        node_list = GraphParser.get_node_properties_labels_id(
                    in_list=neo4j_result.get_data_on_results()
                )

        return Resp.build_success(
            data={
                "nodeList": node_list
            }
        )

    @classmethod
    def add_node(cls, token, chosen_node_label, chosen_node_id, related_node_label, related_text_name):
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        text_id = RandomStrGenerator.generate_uuid()

        neo4j_result = PSBMapper.merge_text_node(
            chosen_node_label=chosen_node_label,
            chosen_node_id=chosen_node_id,
            related_node_label=related_node_label,
            text_id=text_id,
            related_text_name=related_text_name
        )
        if not neo4j_result.check:
            return Resp.build_db_error()

        return Resp.build_success()

    @classmethod
    def translate(cls, token, str_to_translate: str):
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        text_translator_result = TextTranslator.translate(
            target_str=str_to_translate
        )
        if not text_translator_result.check:
            return Resp.build_error()

        translated_str = text_translator_result.get_data_on_results()

        return Resp.build_success(
            data={
                "translatedStr": translated_str
            }
        )

    @classmethod
    def show_pdf(cls, token, file_id):
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        file_path = FileSaver.get_file_path(file_id)

        return Resp.to_show_file(
            file_path=file_path,
            file_type="application/pdf"
        )

    @classmethod
    def get_my_chosen_type_nodes(cls, token, node_label):
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        neo4j_result = PSBMapper.match_all_from_user_node_where_user_id_and_node_label(
            user_id=user_id,
            node_label=node_label
        )
        if not neo4j_result.check:
            return Resp.build_db_error()

        node_name_list = GraphParser.get_properties(
            in_list=neo4j_result.get_data_on_results(),
            name_mapping={NodeIdStr.mapping()[node_label]: "id", NodeNameStr.mapping()[node_label]: "value"}
        )

        return Resp.build_success(
            data={
                "nodeNameList": node_name_list
            }
        )

    @classmethod
    def get_my_all_nodes(cls, token):
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        neo4j_result = PSBMapper.match_all_from_user_node_return_path_max_length(
            user_id=user_id
        )
        if not neo4j_result.check:
            return Resp.build_db_error()

        max_length = neo4j_result.get_data_on_results()[0][0]

        if max_length is None:
            max_length = 1

        neo4j_result1 = PSBMapper.match_user_node(
            user_id=user_id
        )
        if not neo4j_result1.check:
            return Resp.build_db_error()

        neo4j_result2_list = []
        for length in range(max_length):
            neo4j_result2 = PSBMapper.match_all_from_user_node_return_mapping(
                user_id=user_id,
                layer=length+1
            )
            if not neo4j_result2.check:
                return Resp.build_db_error()

            neo4j_result2_list.append(neo4j_result2.get_data_on_results())

        node_list, relation_list = GraphParser.convert_graph_into_node_and_relation(
            n=neo4j_result1.get_data_on_results(),
            n_list=neo4j_result2_list
        )

        node_color, node_edge_color = GraphParser.get_color()

        label_count = GraphParser.count_label_type(
            node_list=node_list
        )

        return Resp.build_success(
            data={
                "nodeList": node_list,
                "relationList": relation_list,
                "nodeColor": node_color,
                "nodeEdgeColor": node_edge_color,
                "labelCount": label_count
            }
        )

    @classmethod
    def upload_file(cls, token, file, file_name: str, file_size: str, file_type: str):
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        uuid = RandomStrGenerator.generate_uuid()

        mysql_result = PSBMapper.transaction_insert_file(
            file_id=uuid,
            file_name=file_name,
            file_size=file_size,
            file_type=file_type,
            owner_id=user_id
        )
        if not mysql_result.check:
            return Resp.build_db_error()

        neo4j_result = PSBMapper.transaction_merge_file_node(
            file_id=uuid,
            file_name=file_name,
            file_size=file_size,
            file_type=file_type,
            owner_id=user_id
        )
        if not neo4j_result.check:
            sql_driver_result = SqlDriver.rollback(mysql_result.get_data_on_results())
            return Resp.build_db_error()

        file_saver_result = FileSaver.save(
            file=file,
            uuid_file_name=uuid
        )

        if not file_saver_result.check:
            sql_driver_result = SqlDriver.rollback(mysql_result.get_data_on_results())
            cypher_driver_result = CypherDriver.rollback(neo4j_result.get_data_on_results())

            return Resp.build_error(
                code=50001,
                message="文件保存失败"
            )

        sql_driver_result = SqlDriver.close(mysql_result.get_data_on_results())
        cypher_driver_result = CypherDriver.close(neo4j_result.get_data_on_results())

        return Resp.build_success()

    @classmethod
    def get_my_files(cls, token):
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        mysql_result = PSBMapper.select_all_from_file_where_owner_id(
            owner_id=user_id
        )
        if not mysql_result.check:
            return Resp.build_db_error()

        return Resp.build_success(
            data={
                "files": FormatConverter.object_list_to_table(
                    list=FormatConverter.list_objects_to_dict(
                        list=mysql_result.get_data_on_results()
                    )
                )
            }
        )
