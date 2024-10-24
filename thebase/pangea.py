from pathlib import Path 
import pandas as pd 
import os 

path = Path("../../IRMAS/IRMAS-TrainingData")

relative_path_rows = []
full_path_rows = []
instrument_rows = []
instrument_id_rows = []

instrument_id = 0 

for innerPath in path.iterdir():
    if os.path.isdir(innerPath):
        instrument = os.path.basename(innerPath)

        count = 0

        for filePath in innerPath.iterdir():
            count += 1

            relative_path_rows.append(filePath)
            full_path_rows.append(filePath.resolve())
            instrument_rows.append(instrument)
            instrument_id_rows.append(instrument_id)


        instrument_id += 1
        print(f"{innerPath} contains {count} samples")

df = pd.DataFrame({
    "relative_path": relative_path_rows,
    "full_path": full_path_rows,
    "instrument": instrument_rows,
    "instrument_id": instrument_id_rows,
})

df.to_csv("data.csv")

