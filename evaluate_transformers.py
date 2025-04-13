import os
import subprocess

# Define the ranges for EPOCHS and MAX_SEQ
epochs_list = [150]
max_seq_list = range(10, 21)

# Loop over the combinations of EPOCHS and MAX_SEQ
for model in [
    "transformer",
]:
    for epochs in epochs_list:
        for max_seq in max_seq_list:
            # Set the environment variables
            os.environ["MODEL_TYPE"] = model
            os.environ["EPOCHS"] = str(epochs)
            os.environ["MAX_SEQ"] = str(max_seq)

            # Call the script with the environment variables
            subprocess.run(["python", "scripts/evaluate_transformer.py"])

            # Print the current setting
            print(
                f"Evaluated with MODEL_TYPE={model} and EPOCHS={epochs} and MAX_SEQ={max_seq}"
            )
