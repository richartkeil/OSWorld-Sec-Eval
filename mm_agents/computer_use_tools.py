def get_script_from_computer_tool(content: dict) -> str:
    tool_input = content.get("input", {})
    action = tool_input.get("action")
    if not action:
        raise ValueError("Action is required")
        
    # Map tool actions to PyAutoGUI commands
    if action == "key":
        key = tool_input.get("text", "")
        # Map common key names
        key_mapping = {
            "Return": "enter",
            "KP_0": "num0",
            # Add more key mappings as needed
        }
        key = key_mapping.get(key, key)
        # Handle key combinations
        if "+" in key:
            return f"import pyautogui; pyautogui.hotkey('{key.replace('+', '\', \'')}')"
        else:
            return f"import pyautogui; pyautogui.press('{key}')"
            
    elif action == "type":
        text = tool_input.get("text", "")
        return f"import pyautogui; pyautogui.write('{text}')"
        
    elif action == "mouse_move":
        coords = tool_input.get("coordinate", [0, 0])
        return f"import pyautogui; pyautogui.moveTo({coords[0]}, {coords[1]})"
        
    elif action == "left_click":
        return "import pyautogui; pyautogui.click()"
        
    elif action == "left_click_drag":
        coords = tool_input.get("coordinate", [0, 0])
        return f"import pyautogui; pyautogui.dragTo({coords[0]}, {coords[1]})"
        
    elif action == "right_click":
        return "import pyautogui; pyautogui.rightClick()"
        
    elif action == "middle_click":
        return "import pyautogui; pyautogui.middleClick()"
        
    elif action == "double_click":
        return "import pyautogui; pyautogui.doubleClick()"
        
    else:
        raise ValueError(f"Unsupported action: {action}")

def get_script_from_bash_tool(content: dict) -> str:
    tool_input = content.get("input", {})
    
    if tool_input.get("restart"):
        raise ValueError("Restart flag is not supported")
    
    command = tool_input.get("command")
    if not command:
        raise ValueError("Command is required unless restarting")
        
    # The environment will execute commands prefixed with "BASH "
    return f"BASH {command}"