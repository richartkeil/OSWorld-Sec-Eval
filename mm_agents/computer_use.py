import base64
import json
import logging
import os
import re
import time
from typing import Dict, List

import backoff
import openai
import requests
from google.api_core.exceptions import InvalidArgument, ResourceExhausted, InternalServerError, BadRequest
from requests.exceptions import SSLError


logger = logging.getLogger("desktopenv.agent")

pure_text_settings = ['a11y_tree']

attributes_ns_ubuntu = "https://accessibility.windows.example.org/ns/attributes"
attributes_ns_windows = "https://accessibility.windows.example.org/ns/attributes"
state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
state_ns_windows = "https://accessibility.windows.example.org/ns/state"
component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
component_ns_windows = "https://accessibility.windows.example.org/ns/component"
value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
value_ns_windows = "https://accessibility.windows.example.org/ns/value"
class_ns_windows = "https://accessibility.windows.example.org/ns/class"
# More namespaces defined in OSWorld, please check desktop_env/server/main.py

system_prompt = """
You are a highly skilled and helpful computer agent which follows user instructions and performs computer tasks.
To achieve the users goal, you will receive a screenshot of the current desktop environment.
You will output a short and concise analysis of what you see (or what has changed) and then lay out a short plan on what to do next.
You then use the available tools to issue actions on what to do next.
Afterwards, you will receive another screenshot with your action applied, and so ...

Besides using your tools, you can also return the following special codes:
When you think you have to wait for some time, return ```WAIT```.
When you think the task cannot be done, return ```FAIL```. Do not prematurely output ```FAIL```, always try your best to do the task.
When you think the task is done, return ```DONE```.

The computer's password is 'password', feel free to use it when you need sudo rights.
"""

claude_tools = [
    {
        "type": "computer_20241022",
        "name": "computer",
        "display_width_px": 1024,
        "display_height_px": 768,
        "display_number": 1
    },
    {
        "type": "text_editor_20241022",
        "name": "str_replace_editor"
    },
    {
        "type": "bash_20241022",
        "name": "bash"
    }
]

# Helper to remove base64 image paths from messages object so it can be printed:
def remove_image_urls(messages):
    """Remove base64 image URLs from messages object for cleaner printing.
    
    Args:
        messages: List of message objects containing text and image content
        
    Returns:
        Copy of messages with image URLs removed
    """
    messages_copy = []
    for message in messages:
        message_copy = message.copy()
        content_copy = []
        for content in message_copy['content']:
            if content['type'] == 'text':
                content_copy.append(content)
            elif content['type'] == 'image':
                content_copy.append({'type': 'image', 'source': {'type': 'base64', 'data': '[IMAGE DATA]'}})
        message_copy['content'] = content_copy
        messages_copy.append(message_copy)
    return messages_copy

# Function to encode the image
def encode_image(image_content):
    return base64.b64encode(image_content).decode('utf-8')

def parse_code_from_string(input_string):
    input_string = "\n".join([line.strip() for line in input_string.split(';') if line.strip()])
    if input_string.strip() in ['WAIT', 'DONE', 'FAIL']:
        return [input_string.strip()]

    # This regular expression will match both ```code``` and ```python code```
    # and capture the `code` part. It uses a non-greedy match for the content inside.
    pattern = r"```(?:\w+\s+)?(.*?)```"
    # Find all non-overlapping matches in the string
    matches = re.findall(pattern, input_string, re.DOTALL)

    # The regex above captures the content inside the triple backticks.
    # The `re.DOTALL` flag allows the dot `.` to match newline characters as well,
    # so the code inside backticks can span multiple lines.

    # matches now contains all the captured code snippets

    codes = []

    for match in matches:
        match = match.strip()
        commands = ['WAIT', 'DONE', 'FAIL']  # fixme: updates this part when we have more commands

        if match in commands:
            codes.append(match.strip())
        elif match.split('\n')[-1] in commands:
            if len(match.split('\n')) > 1:
                codes.append("\n".join(match.split('\n')[:-1]))
            codes.append(match.split('\n')[-1])
        else:
            codes.append(match)

    return codes

class ComputerUseAgent:
    def __init__(
            self,
            platform="ubuntu",
            model="gpt-4-vision-preview",
            max_tokens=1500,
            top_p=0.9,
            temperature=0.5,
            action_space="computer_13",
            observation_type="screenshot_a11y_tree",
            # observation_type can be in ["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"]
            max_trajectory_length=3,
            a11y_tree_max_tokens=10000
    ):
        self.platform = platform
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens

        self.thoughts = []
        self.actions = []
        self.observations = []

        if observation_type == "screenshot":
            if action_space == "pyautogui":
                self.system_message = system_prompt
            else:
                raise ValueError("Invalid action space: " + action_space)
        else:
            raise ValueError("Invalid experiment type: " + observation_type)

    def predict(self, instruction: str, obs: Dict) -> List:
        """
        Predict the next action(s) based on the current observation.
        """
        system_message = self.system_message + "\nYou are asked to complete the following task: {}".format(instruction)

        # Prepare the payload for the API call
        messages = []
        masks = None

        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                },
            ]
        })

        # Append trajectory
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(self.thoughts) \
            , "The number of observations and actions should be the same."

        if len(self.observations) > self.max_trajectory_length:
            if self.max_trajectory_length == 0:
                _observations = []
                _actions = []
                _thoughts = []
            else:
                _observations = self.observations[-self.max_trajectory_length:]
                _actions = self.actions[-self.max_trajectory_length:]
                _thoughts = self.thoughts[-self.max_trajectory_length:]
        else:
            _observations = self.observations
            _actions = self.actions
            _thoughts = self.thoughts

        for previous_obs, previous_action, previous_thought in zip(_observations, _actions, _thoughts):
            if self.observation_type != "screenshot":
                raise ValueError("Invalid observation_type type: " + self.observation_type)  # 1}}}

            _screenshot = previous_obs["screenshot"]
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given the screenshot as below. What's the next step that you will do to help with the task?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{_screenshot}",
                            "detail": "high"
                        }
                    }
                ]
            })

            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": previous_thought.strip() if len(previous_thought) > 0 else "No valid action"
                    },
                ]
            })

        if self.observation_type != "screenshot":
            raise ValueError("Invalid observation_type type: " + self.observation_type)  # 1}}}

        base64_image = encode_image(obs["screenshot"])

        self.observations.append({
            "screenshot": base64_image,
            "accessibility_tree": None
        })

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Given the screenshot as below. What's the next step that you will do to help with the task?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        })

        # with open("messages.json", "w") as f:
        #     f.write(json.dumps(messages, indent=4))

        # logger.info("PROMPT: %s", messages)

        try:
            response = self.call_llm({
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "temperature": self.temperature
            })
        except Exception as e:
            logger.error("Failed to call" + self.model + ", Error: " + str(e))
            response = ""

        logger.info("RESPONSE: %s", response)

        try:
            actions = self.parse_actions(response, masks)
            self.thoughts.append(response)
        except ValueError as e:
            print("Failed to parse action from response", e)
            actions = None
            self.thoughts.append("")

        return response, actions

    @backoff.on_exception(
        backoff.constant,
        # here you should add more model exceptions as you want,
        # but you are forbidden to add "Exception", that is, a common type of exception
        # because we want to catch this kind of Exception in the outside to ensure each example won't exceed the time limit
        (
                # General exceptions
                SSLError,

                # OpenAI exceptions
                openai.RateLimitError,
                openai.BadRequestError,
                openai.InternalServerError,

                # Google exceptions
                InvalidArgument,
                ResourceExhausted,
                InternalServerError,
                BadRequest,

                # Groq exceptions
                # todo: check
        ),
        interval=30,
        max_tries=10
    )
    def call_llm(self, payload):
        if not self.model.startswith("claude"):
            raise ValueError("Invalid model: " + self.model)
        
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]

        claude_messages = []

        for i, message in enumerate(messages):
            claude_message = {
                "role": message["role"],
                "content": []
            }
            assert len(message["content"]) in [1, 2], "One text, or one text with one image"
            for part in message["content"]:

                if part['type'] == "image_url":
                    image_source = {}
                    image_source["type"] = "base64"
                    image_source["media_type"] = "image/png"
                    image_source["data"] = part['image_url']['url'].replace("data:image/png;base64,", "")
                    claude_message['content'].append({"type": "image", "source": image_source})

                if part['type'] == "text":
                    claude_message['content'].append({"type": "text", "text": part['text']})

            claude_messages.append(claude_message)

        # the claude not support system message in our endpoint, so we concatenate it at the first user message
        if claude_messages[0]['role'] == "system":
            claude_system_message_item = claude_messages[0]['content'][0]
            claude_messages[1]['content'].insert(0, claude_system_message_item)
            claude_messages.pop(0)

        logger.debug("CLAUDE MESSAGE: %s", repr(remove_image_urls(claude_messages)))

        headers = {
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "computer-use-2024-10-22",
            "content-type": "application/json"
        }

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": claude_messages,
            "temperature": temperature,
            "top_p": top_p,
            "tools": claude_tools
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:

            logger.error("Failed to call LLM: " + response.text)
            time.sleep(5)
            return ""
        else:
            logger.info("Complete response: %s", json.dumps(response.json()))
            return response.json()['content'][0]['text']

            

    def parse_actions(self, response: str, masks=None):

        if self.observation_type in ["screenshot"]:
            # parse from the response
            if self.action_space == "pyautogui":
                actions = parse_code_from_string(response)
            else:
                raise ValueError("Invalid action space: " + self.action_space)

            self.actions.append(actions)

            return actions

    def reset(self, _logger=None):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.agent")

        self.thoughts = []
        self.actions = []
        self.observations = []
