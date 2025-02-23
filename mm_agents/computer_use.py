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

from mm_agents.computer_use_tools import get_script_from_bash_tool, get_script_from_computer_tool


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

# ANTRHOPIC PROMPT
#
# <SYSTEM_CAPABILITY>
# * You are utilising an Ubuntu virtual machine using aarch64 architecture with internet access.
# * You can feel free to install Ubuntu applications with your bash tool. Use curl instead of wget.
# * To open firefox, please just click on the firefox icon.  Note, firefox-esr is what is installed on your system.
# * Using bash tool you can start GUI applications, but you need to set export DISPLAY=:1 and use a subshell. For example "(DISPLAY=:1 xterm &)". GUI apps run with bash tool will appear within your desktop environment, but they may take some time to appear. Take a screenshot to confirm it did.
# * When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_editor or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
# * When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
# * When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
# * The current date is Wednesday, October 23, 2024.
# </SYSTEM_CAPABILITY>

# <IMPORTANT>
# * When using Firefox, if a startup wizard appears, IGNORE IT.  Do not even click "skip this step".  Instead, click on the address bar where it says "Search or enter address", and enter the appropriate search term or URL there.
# * If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your StrReplaceEditTool.
# </IMPORTANT>

system_prompt = """
You are a highly skilled and helpful computer agent which follows user instructions and performs computer tasks.
To achieve the users goal, you will receive a screenshot of the current desktop environment.
You will output a short and concise analysis of what you see (or what has changed).
You then use the available tools to issue actions on what to do next. You can output multiple tool calls in one response, and they will be executed in order.
If unsure, better wait until you receive the next screenshot to confirm that your action was successful.
After each response, you will receive another screenshot with your action applied, and so on...

Besides using your tools, you can also return the following special codes:
When you think you have to wait for some time, return ```WAIT```.
When you think the task cannot be done, return ```FAIL```. Do not prematurely output ```FAIL```, always try your best to do the task.
When you think the task is done, return ```DONE```.

For the tool computer_20241022, DO NOT use the actions "screenshot" and "cursor_position", as you will always receive a screenshot.
DO NOT use the tool text_editor_20241022 at all. For modifying files, use the tool bash_20241022.

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
    if "WAIT" in input_string:
        return ["WAIT"]
    if "DONE" in input_string:
        return ["DONE"]
    if "FAIL" in input_string:
        return ["FAIL"]
    return []

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
            _shell_output = previous_obs["shell_output"]
            _shell_exit_code = previous_obs["shell_exit_code"]

            if _shell_output:
                if _shell_exit_code == 0 and _shell_output.strip() == "":
                    hint = "The command completed successfully and produced no output."
                else:
                    hint = f"The command completed with exit code {_shell_exit_code}.\nThe output is:\n{_shell_output}"

                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": hint}
                    ]
                })


            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the next screenshot."
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

        _shell_output = obs["shell_output"]
        _shell_exit_code = obs["shell_exit_code"]
        if _shell_exit_code is not None:
            if _shell_exit_code == 0 and _shell_output.strip() == "":
                hint = "The command completed successfully and produced no output."
            else:
                hint = f"The command completed with exit code {_shell_exit_code}.\nThe output is:\n{_shell_output}"

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": hint}
                ]
            })
        

        base64_image = encode_image(obs["screenshot"])

        self.observations.append({
            "screenshot": base64_image,
            "shell_output": _shell_output,
            "shell_exit_code": _shell_exit_code
        })

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the next screenshot."
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

        responseText = response["content"][0]["text"]

        logger.info(f"CLAUDE SAYS:\n{responseText}")

        try:
            if response["stop_reason"] == "tool_use":
                actions = self.parse_tool_use(response)
            else:
                actions = self.parse_text(responseText)
            self.thoughts.append(responseText)
        except ValueError as e:
            logger.error(f"Failed to parse action from response: {e}")
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

        # logger.debug("CLAUDE MESSAGE: %s", repr(remove_image_urls(claude_messages)))

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
            return response.json()


    def parse_tool_use(self, response: Dict):
        """
        Maps Claude's tool use actions to PyAutoGUI commands.
        
        Args:
            response (Dict): Response from Claude containing tool use information
            
        Returns:
            List[str]: List of PyAutoGUI commands to execute
        """
        if "content" not in response:
            return []
            
        actions = []
        try:
            # Look for all tool invocations in the response
            for content in response["content"]:
                if content.get("type") != "tool_use":
                    continue

                logger.info(f"CLAUDE WANTS TO DO:\n{json.dumps(content.get('input'))}")

                if content.get("name") == "computer":
                    action = get_script_from_computer_tool(content)
                elif content.get("name") == "bash":
                    action = get_script_from_bash_tool(content)
                else:
                    raise ValueError(f"Unsupported tool: {content.get('name')}")
                    
                actions.append(action)

            self.actions.append(actions)
            
            return actions
            
        except Exception as e:
            logger.error(f"Error parsing tool use action: {str(e)}")
            return []
            

    def parse_text(self, response: str, masks=None):
        # parse from the response
        actions = parse_code_from_string(response)

        self.actions.append(actions)

        return actions

    def reset(self, _logger=None):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.agent")

        self.thoughts = []
        self.actions = []
        self.observations = []
