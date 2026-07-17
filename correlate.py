"""
Reads combined_output.txt and, for every 'no connection' warning line,
prints the most recent preceding DEBUG line (the episode/task that was
active when the warning occurred).
"""
import sys
import re
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "combined_output.txt"

last_debug = None
task_counts = Counter()
examples_shown = 0

with open(path, errors="replace") as f:
    for line in f:
        if line.startswith("[DEBUG"):
            last_debug = line.strip()
        elif "no connection" in line:
            if last_debug:
                # extract task=... from the debug line
                m = re.search(r"task=(\w+)", last_debug)
                task = m.group(1) if m else "UNKNOWN"
                task_counts[task] += 1
                if examples_shown < 10:
                    print(f"WARNING preceded by: {last_debug}")
                    examples_shown += 1
            else:
                task_counts["NO_PRIOR_DEBUG_LINE"] += 1

print()
print("=== Summary: which task was active when 'no connection' warnings occurred ===")
for task, count in task_counts.most_common():
    print(f"  {task}: {count} warnings")