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
    print(parse_action(inp))
