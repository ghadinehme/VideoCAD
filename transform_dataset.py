"""
Write a script that checks for each "move to" command, if it is followed by a "click" command.
If not, then print to console what it is followed by
"""

import re
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
import os
import json
from utils import *
import pickle as pkl

# --- Constants ---
SCALE_FACTOR = 1000
DEFAULT_VECTOR = [-1] * 7  # Each action is a 10d vector
# Default 10d vector (indices 0..9 all set to -1)
DEFAULT_VECTOR = [-1] * 7

"""
Get rid of:
hold shift
shift+y

condense
enter
up key
left key
down key
right key


esc -> before each curve command press escape

"""

KEY_MAP = {
    'a' : 0,
    'l' : 1,
    'c' : 2,
    'y' : 3,
    'tab' : 4,
    'space': 5,
    'enter' : 6,
    'up' : 7,
    'down' : 8,
    'left' : 9,
    'right' : 10,
    'esc' : 11,
    ('shift', 's') : 12,
    ('shift', 'e') : 13,
    ('shift', 'y') : 14,
    ('shift', 'h') : 15,
    ('shift', 'p') : 16,
    ('shift', 0) : 17,
    ('shift', 1): 18,
    ('shift', '7') : 19,
}

BIN = 50

# Base abstract handler for a log dictionary
class LogActionHandler(ABC):

    def matches_general(self, log: dict) -> bool:
        if log.get("status", "") != "finished":
            return False
        return self.matches(log)

    @abstractmethod
    def matches(self, log: dict) -> bool:
        """Return True if the log entry should be handled by this handler."""
        pass

    @abstractmethod
    def process(self, log: dict, next_log: dict = None):
        """
        Process the log entry (and optionally the next log) to produce a vector and a timestamp.
        """
        pass

# --- Handler Implementations ---

# For "move to" actions.
class MoveToHandler(LogActionHandler):
    def matches(self, log: dict) -> bool:
        # We process only "started" move to entries.
        return (log.get("action", "").lower() == "move to" and 
                log.get("status", "").lower() == "finished")
    
    def process(self, log: dict, next_log: dict = None):
        vector = DEFAULT_VECTOR.copy()
        vector[0] = 0  # We assign type 5 for move to (since type mapping doesn't cover it)
        # Look ahead to the next log for the "finished" move to relative coordinates.
        if (next_log and log.get("action", "").lower() == "move to" and
            log.get("status", "").lower() == "finished"):
            rel = log.get("relative", {})
            vector[1] = round(rel.get("x", -1) * SCALE_FACTOR) + 15
            vector[2] = round(rel.get("y", -1) * SCALE_FACTOR)
        timestamp = log["timestamp"]
        return vector, timestamp

# For "click" actions.
class ClickHandler(LogActionHandler):
    def matches(self, log: dict) -> bool:
        return log.get("action", "").lower() == "click" and log.get("status", "").lower() == "finished"
    
    def process(self, log: dict, next_log: dict = None):
        vector = DEFAULT_VECTOR.copy()
        vector[0] = 4  # Type code for Click
        # No additional modifications for a click
        timestamp = log["timestamp"]
        return vector, timestamp

# For "press keys" or "hotkey" actions.
class PressKeysHandler(LogActionHandler):
    def matches(self, log: dict) -> bool:
        action = log.get("action", "").lower()
        return action in ["press keys", "hotkey"]
    
    def process(self, log: dict, next_log: dict = None):
        vector = DEFAULT_VECTOR.copy()
        vector[0] = 1  # Type code for Press keys / Hotkey
        args = log.get("args", [])
        if not args:
            return False, False
        for k in KEY_MAP.keys():
            if isinstance(k, str):
                if k in args and 'shift' not in args:
                    vector[3] = KEY_MAP[k]*BIN
            else:
                if k[0] in args and k[1] in args:
                    vector[3] = KEY_MAP[k]*BIN
        if len(args)>2:
            vector[4] = (int(args[1])-2)*200

        if vector[3] == -1:
            return False, False
        timestamp = log["timestamp"]
        return vector, timestamp
    
class ScrollHandler(LogActionHandler):
    def matches(self, log: dict) -> bool:
        return (log.get("action", "").lower() == "scroll" and 
                log.get("status", "").lower() == "finished")
    
    def process(self, log: dict, next_log: dict = None):
        vector = DEFAULT_VECTOR.copy()
        vector[0] = 2
        args = log.get("args", [])
        if not args:
            return False, False
        vector[5] = (int(args[0])>0)*500
        timestamp = log["timestamp"]
        return vector, timestamp


# For "write text" (or "type") actions.
class WriteTextHandler(LogActionHandler):
    def matches(self, log: dict) -> bool:
        return log.get("action", "").lower() in ["write text", "type"]
    
    def process(self, log: dict, next_log: dict = None):
        vector = DEFAULT_VECTOR.copy()
        vector[0] = 3  # Type code for Write text / Type
        args = log.get("args", [])
        # If the first argument can be converted to float, store it at index 1.
        if args:
            scale = log.get("scale", 0)
            vector[6] = max(min(int(float(args[0])/scale*499) + 500,999),0) # to be checked
        timestamp = log["timestamp"]
        return vector, timestamp

# For "key down" actions.
class KeyDownHandler(LogActionHandler):
    def matches(self, log: dict) -> bool:
        return log.get("action", "").lower() == "key down"
    
    def process(self, log: dict, next_log: dict = None):
        vector = DEFAULT_VECTOR.copy()
        vector[0] = 1
        args = log.get("args", [])
        if "shift" in args:
            vector[3] = BIN*18
        timestamp = log["timestamp"]
        return vector, timestamp

# For "key up" actions.
class KeyUpHandler(LogActionHandler):
    def matches(self, log: dict) -> bool:
        return log.get("action", "").lower() == "key up"
    
    def process(self, log: dict, next_log: dict = None):
        vector = DEFAULT_VECTOR.copy()
        vector[0] = 1
        args = log.get("args", [])
        if "shift" in args:
            vector[3] = BIN*17
        timestamp = log["timestamp"]
        return vector, timestamp

# --- Register the Handlers ---
handlers = [
    ClickHandler(),
    PressKeysHandler(),
    WriteTextHandler(),
    ScrollHandler(),
    KeyUpHandler(),
    KeyDownHandler(),
    MoveToHandler()
]

# --- Main Function to Convert Logs ---
def convert_logs_to_vectors(logs: list, enable_logging=False):
    """
    Given a list of log dictionaries, convert each to a 10d vector
    according to the rules and return a list of vectors and a list of timestamps.
    """
    vectors = []
    timestamps = []
    i = 1
    scale = logs[0].get("scale", 0)
    print(scale)
    while i < len(logs):
        log = logs[i]
        log["scale"] = scale

        handled = False
        for handler in handlers:
            if handler.matches_general(log):
                # For "move to", pass the next log entry if available.
                next_log = logs[i+1] if i+1 < len(logs) else None
                vec, ts = handler.process(log, next_log)
                if not vec:
                    handled = False
                    break
                if vec[0] == 2:
                    if vectors[-1][0] == 2:
                        handled = True
                        timestamps[-1] = ts
                        break
                vectors.append(vec)
                timestamps.append(ts)
                handled = True
                break
        if not handled:
            if enable_logging:
                # For unseen actions, you might want to log or print them.
                if log.get("action") != "move to":
                    print("Unseen action:", log.get("action"), "args:", log.get("args"))
        i += 1
    return vectors, timestamps



def parse_log_line(line : str):
    result = {}
    line = line.strip()

    def match_regex(regex):
        res = re.search(regex, line)
        if not res:
            raise Exception(f"Invalid formatting found: {line}")
        return res
    scale_match = re.search(r"Scale:\s*([\d.]+)", line)
    
    if scale_match:
        result["scale"] = float(scale_match.group(1))
        return result
    
    # 1. Extract timestamp (e.g., "1.003")
    timestamp_match = match_regex(r"(\d+)\s+-\s+INFO")
    result["timestamp"] = float(timestamp_match.group(1)) 

    # 2. Extract message (text between "- INFO - " and the first period)
    message_match = re.search(r"- INFO - (.+?)\.", line)
    if not message_match:
        raise Exception("Invalid formatting found")
    message = message_match.group(1).strip()
    tmp = message.split(" ")
    status, action = tmp[0], " ".join(tmp[1:])
    result["status"] = status
    result["action"] = action
        

    # 3. Extract absolute position: e.g., "Absolute (Point(x=240, y=325))"
    abs_match = match_regex(r"Absolute \(Point\(x=(\d+), y=(\d+)\)\)")
    result["absolute"] = {"x": int(abs_match.group(1)), "y": int(abs_match.group(2))}

    # 4. Extract relative position: e.g., "Relative ((0.09854014598540146, 0.2950191570881226))"
    rel_match = match_regex(r"Relative \(\((-?[\d\.]+), (-?[\d\.]+)\)\)")
    result["relative"] = {'x': float(rel_match.group(1)), 'y': float(rel_match.group(2))}

    # 5. Extract args:
    #    The args portion is like: "Args (<io_env.io_env.IOEnv object at ...>, 'l')"
    #    We'll split by commas and ignore any argument that starts with '<' and ends with '>'
    args_match = match_regex(r"Args \((.+)\)")
    args_str = args_match.group(1)
    # Split the arguments by comma
    args_list = [arg.strip() for arg in args_str.split(",")]
    # Filter out any argument that looks like an object representation
    filtered_args = [arg for arg in args_list if not (arg.startswith("<") and arg.endswith(">"))]
    # Remove quotes from the remaining arguments
    filtered_args = [arg.strip("'\"[]") for arg in filtered_args]
    result["args"] = filtered_args

    
    return result


def process_logs(log_text):
    lines = log_text.strip().splitlines()
    res = []
    for line in lines:
        res.append(parse_log_line(line))
    return res

def process_logs_filtered(log_text):
    lines = log_text.strip().splitlines()
    res = []
    for line in lines:
        log_line = parse_log_line(line)
        for handler in handlers:
            if handler.matches_general(log_line):
                res.append(log_line)
                break
    
    return res


def process_logs(log_text, is_filtered=False):
    lines = log_text.strip().splitlines()
    res = []
    for line in lines:
        log_line = parse_log_line(line)
        if not is_filtered:
            res.append(log_line)
            continue
        for handler in handlers:
            if handler.matches_general(log_line):
                res.append(log_line)
                break
    return res

def mouse_log_to_dict(source_dir, target_dir, is_filtered=False):
    files = os.listdir(source_dir)
    for file in tqdm(files):
        log_text = open_file(os.path.join(source_dir, file))
        logs = process_logs(log_text, is_filtered)
        file_name = os.path.splitext(file)[0]
        output_file = os.path.join(target_dir, f"{file_name}.json")
        save_json(logs, output_file)



def dict_to_vec(source_dir, target_dir):
    files = os.listdir(source_dir)
    for file in tqdm(files):
        js = load_json(os.path.join(source_dir, file))
        file_name = os.path.splitext(file)[0]
        
        output_file = os.path.join(target_dir, f"{file_name}.pkl")
        if os.path.exists(output_file):
            continue
        actions, timesteps = convert_logs_to_vectors(js)
        

        actions = np.array(actions)
        timesteps = np.array(timesteps)

        end_action = 950
        end_idx = np.where(actions[:, 3] == end_action)[0]
        if len(end_idx) > 0:
            print(f"End idx: {end_idx}")

            actions = actions[:end_idx[0]+1]
            timesteps = timesteps[:end_idx[0]+1]
            np.savetxt(os.path.join(target_dir, f"{file_name}.csv"), actions, delimiter=",", fmt="%s")
            np.savetxt(os.path.join(target_dir, f"{file_name}_t.csv"), timesteps, delimiter=",", fmt="%s")
            with open(output_file, "wb") as file:
                pkl.dump((actions, timesteps), file)


# --- Example Usage ---
if __name__ == "__main__":
    mouse_log_to_dict("data_loader/test_dataset/mouse", "data_loader/test_dataset/mouse_json_filtered")
    dict_to_vec("data_loader/test_dataset/mouse_json_filtered", "data_loader/test_dataset/vec")
    print("Hello")