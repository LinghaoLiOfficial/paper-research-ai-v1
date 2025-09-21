from textblob import TextBlob


class NLPAlgorithm:
    # 提取专有名词
    @classmethod
    def extract_NNP_entity(cls, text):
        blob = TextBlob(text)

        phrases = blob.noun_phrases  # 提取名词短语
        keywords = [
            phrase for phrase in phrases
            if any(word.istitle() for word in phrase.split())  # 包含大写词
               or any(c in phrase for c in '-/()')  # 包含特殊符号
        ]

        return keywords
