import logging

import great_expectations as gx
from great_expectations.core import ExpectationConfiguration

from src.voicegraph.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_gx_suite():
    context = gx.get_context()
    suite_name = settings.gx_expectation_suite_name
    suite = context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

    expectations = [
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "user_id"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "consent_to_call"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={"column": "consent_to_call", "type_": "bool"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_match_regex",
            kwargs={"column": "phone_hash", "regex": "^[a-f0-9]{64}$"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "ltv_segment", "value_set": ["PREMIUM", "STANDARD", "LOW"]},
        ),
    ]

    for exp in expectations:
        suite.add_expectation(exp)

    context.save_expectation_suite(suite)
    context.build_data_docs()
    logger.info(f"Expectation Suite '{suite_name}' успешно создан и сохранен.")


if __name__ == "__main__":
    initialize_gx_suite()
