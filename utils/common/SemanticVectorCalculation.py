from utils.llm.APILLMParser import APILLMParser


class SemanticVectorCalculation:

    # 获取每个主题词的高维特征向量
    @classmethod
    def calculate_text_group(cls, text_list, group_num=50):

        text_num = len(text_list)
        send_turn = text_num // group_num
        last_text_num = text_num % group_num
        text_num_list = [group_num] * send_turn + [last_text_num] if last_text_num != 0 else [group_num] * send_turn

        total_embedding_list = []
        for i, keyword_num in enumerate(text_num_list):
            before_num = 0 if i == 0 else sum(text_num_list[:i])
            current_text_list = text_list[before_num: before_num + keyword_num]
            embedding_list = APILLMParser.call_embedding_llm_api(current_text_list)
            total_embedding_list = total_embedding_list + embedding_list

        return total_embedding_list
