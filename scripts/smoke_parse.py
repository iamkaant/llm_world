import os
import sys

sys.path.append(os.path.join(os.getcwd(), "src"))

from main import parse_action


def main() -> None:
    inputs = [
        '{"next_action":"move north"}',
        "Next Action: y+1",
        "I will go east",
    ]

    for idx, raw in enumerate(inputs, start=1):
        action = parse_action(raw)
        print(f"{idx}. input={raw!r} -> next_action={action.next_action}")


if __name__ == "__main__":
    main()
