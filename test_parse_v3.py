import sys
import os

# Add src to sys.path to ensure we can import from it
sys.path.append(os.path.join(os.getcwd(), 'src'))

from main import parse_action

inputs = [
    '{"next_action":"move north"}',
    'Next Action: y+1',
    'I will go east'
]

for inp in inputs:
    try:
        action = parse_action(inp)
        print(action.next_action)
    except Exception as e:
        print(f"Error: {e}")
