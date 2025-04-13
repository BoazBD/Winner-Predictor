import os
import subprocess

# Define the ranges for EPOCHS and MAX_SEQ
epochs_list = [100]
max_seq_list = range(10,15)
thresholds = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]

# Loop over the combinations of EPOCHS and MAX_SEQ
for model in [
    "lstm",
]:
    for epochs in epochs_list:
        for max_seq in max_seq_list:
            for threshold in thresholds:
                # Set the environment variables
                os.environ["MODEL_TYPE"] = model
                os.environ["EPOCHS"] = str(epochs)
                os.environ["MAX_SEQ"] = str(max_seq)
                os.environ["THRESHOLD"] = str(threshold)

                # Call the script with the environment variables
                subprocess.run(["python", "scripts/evaluate_model_on_last_week.py"])

                # Print the current setting
                print(f"Evaluated with MODEL_TYPE={model} and EPOCHS={epochs} and MAX_SEQ={max_seq}")
