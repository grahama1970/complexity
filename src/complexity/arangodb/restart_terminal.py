import sys
import os

orig_stdout = sys.stdout  # Save the original stdout
sys.stdout = open(os.devnull, 'w')  # Redirect to null

try:
    # Your code here
    print("This won't show")
finally:
    sys.stdout.close()  # Close the null device
    sys.stdout = orig_stdout  # Restore original stdout