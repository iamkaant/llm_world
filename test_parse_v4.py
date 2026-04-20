import sys
import os
import json
import re
from dataclasses import dataclass
from typing import Optional

# Mocking the Action class if necessary or just importing it
# But we can just import the whole thing safely
sys.path.append(os.path.join(os.getcwd(), 'src'))
from main import parse_action

inputs = [
    '{"next_action":"move north"}',
    'Next Action: y+1',
    'I will go east'
]

for i, inp in enumerate(inputs):
    # print(f"Input {i}: {inp}")
    try:
        action = parse_action(inp)
        print(action.next_action)
    except Exception as e:
        print(f"Error on input {i}: {e}")
