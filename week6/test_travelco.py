from custom_eval import Custom_GeminiPro

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

from deepeval import evaluate


custom_llm = Custom_GeminiPro()
answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=custom_llm)

def test_travelco_oneshot():
    test_agra_safety = LLMTestCase(
        input="Is Agra a safe place?",
        expected_output="You have to be careful as a tourist about theft and pickpockets.",
        actual_output=(
            "Staying safe in Agra requires being mindful of petty theft."
            " Be sure to keep valuables secure, avoid paying anyone upfront"
            " (including taxi drivers), and be aware of pickpockets."
            " When shopping, remember that many items are cheap replicas,"
            " and be wary of scams like the 'bait and switch' – especially"
            " at shops like Kanu Carpets. It's advisable to carry your purchases"
            " with yourself. During winter, pack for unpredictable weather."
        ),
    )


    test_bangalore_places = LLMTestCase(
        input="What are some good places to see in Bangalore?",
        expected_output="Sorry, I don't have any travel info about Bangalore",
        actual_output=(
            "I'm sorry, but the provided text focuses on Jaipur, Mumbai,"
            " Delhi, and Agra, and doesn't contain any information about"
            " Bangalore attractions. I would need a different travel guide"
            " to help you with that."
        ),
    )


    test_family_beaches = LLMTestCase(
        input = "What are some good family friendly beaches?",
        expected_output="Hawaii has a great family friendly and safe beach",
        actual_output="""
            For family-friendly beach fun, consider these options:
            Ala Moana Park in Honolulu: A local favorite with calm waters, perfect for families and a relaxing swim. It's a great alternative to the more crowded Waikiki Beach.
            Repulse Bay in Hong Kong: This large urban beach has amenities that are particularly appealing to families with young children.
            Manhattan Beach, Hermosa Beach, and Redondo Beach in Los Angeles: These beaches in the South Bay are popular with families and offer piers and expansive sandy areas.
            In Toronto, many beaches along the Great Lakes are within a short driving distance and ideal for summer days. The guide mentions Wasaga, Sauble Beach, Sandbanks, Grand Bend, Long Point, and Turkey Point.
            Remember to always check safety guidelines and lifeguard presence at any beach you choose.
            """
    )
    

test_cases = [test_agra_safety, test_bangalore_places, test_family_beaches]
answer_metrics = [answer_relevancy_metric]
# assert_test(test_cases, answer_metrics)
evaluation_results = evaluate(test_cases, answer_metrics)
print(evaluation_results)

