from .base import LMAgent
import openai
import logging
import traceback
from mint.datatypes import Action
import backoff
from numbers import Number

LOGGER = logging.getLogger("MINT")


class OpenAILMAgent(LMAgent):
    def __init__(self, config):
        super().__init__(config)
        assert "model_name" in config.keys()

    @backoff.on_exception(
        backoff.fibo,
        # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        (
            openai.error.APIError,
            openai.error.Timeout,
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
            openai.error.APIConnectionError,
        ),
    )
    def call_lm(self, messages):
        # Prepend the prompt with the system message
        response = openai.ChatCompletion.create(
            model=self.config["model_name"],
            messages=messages,
            max_tokens=self.config.get("max_tokens", 512),
            temperature=self.config.get("temperature", 0),
            stop=self.stop_words,
        )
        return response.choices[0].message["content"], response["usage"]

    def update_token_counter(self, token_counter, token_usage, prefix=""):
        for usage_type, count in token_usage.items():
            # LM Studio may include nested usage detail objects such as
            # completion_tokens_details; only aggregate plain numeric counters.
            if isinstance(count, Number):
                token_counter[prefix + usage_type] += int(count)

    def act(self, state):
        messages = state.history
        try:
            lm_output, token_usage = self.call_lm(messages)
            self.update_token_counter(state.token_counter, token_usage)
            action = self.lm_output_to_action(lm_output)
            return action
        except openai.error.InvalidRequestError:  # mostly due to model context window limit
            tb = traceback.format_exc()
            return Action(f"", False, error=f"InvalidRequestError\n{tb}")
        # except Exception as e:
        #     tb = traceback.format_exc()
        #     return Action(f"", False, error=f"Unknown error\n{tb}")
