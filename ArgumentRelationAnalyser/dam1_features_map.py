

from src.decompose.get_components import FunctionalComponentsExtractor
from src.features.sentiment import get_sentiment
from src.features.similarity import get_sim_dam1_2, sim_feature

class Dam1ArgumentRelationAnalyzer:
    def __init__(self):
        pass

    @classmethod
    def get_argument_relation(cls, p1p2):
        text1, text2 = p1p2
        functional_components_p1, functional_components_p2 = cls._get_functional_components(text1, text2)
        sentiment = cls._get_sentiment(text1, text2)
        arg_rel1 = cls._get_arg_relation(functional_components_p1, functional_components_p2, sentiment)
        arg_rel2 = cls._get_arg_relation(functional_components_p2, functional_components_p1, sentiment)
        return cls._final_result(arg_rel2, arg_rel1)

    @classmethod
    def _get_arg_relation(cls, components1, components2, sentiment):
        similarity, antonymy = cls._get_sim(components1, components2)
        if not antonymy:
            antonymy = [0]
        return cls._sim_entail_argrel(similarity, sentiment, antonymy)

    @staticmethod
    def _sim_entail_argrel(similarity, sentiment, antonymy):
        if sim_feature(similarity) and not sentiment:
            return "Attack"
        elif antonymy[0] == 1 and (not sentiment):
            return "Attack"
        elif sim_feature(similarity) and sentiment and antonymy[0] == 0:
            return "Inference"
        else:
            return "None"

    @staticmethod
    def _final_result(arg_rel2, arg_rel1):
        if arg_rel2 == "Attack" or arg_rel1 == "Attack":
            return "CA"
        elif arg_rel2 == "Inference" or arg_rel1 == "Inference":
            return "RA"
        else:
            return "None"

    @staticmethod
    def _get_functional_components(text1, text2):
        extractor = FunctionalComponentsExtractor()
        return extractor.get_rule_based_functional_components((text1, text2))

    @staticmethod
    def _get_sentiment(text1, text2):
        return get_sentiment(text1, text2)

    @staticmethod
    def _get_sim(components1, components2):
        return get_sim_dam1_2(components1, components2)


