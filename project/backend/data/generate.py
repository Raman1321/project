import numpy as np
import pandas as pd
from core.config import DATA_PATH

def generate_data(n=5000):
    np.random.seed(42)

    df = pd.DataFrame({
        "age": np.random.randint(18, 65, n),
        "gender": np.random.choice(["Male", "Female"], n),
        "caste": np.random.choice(["General", "OBC", "SC", "ST"], n),
        "income": np.random.randint(20000, 300000, n),
        "region": np.random.choice(["Urban", "Rural"], n),
        "education_years": np.random.randint(0, 16, n),
        "employment": np.random.choice([0, 1], n)
    })

    df["eligibility"] = (
        (df["income"] < 100000) &
        (df["education_years"] >= 5) &
        (df["employment"] == 1)
    ).astype(int)

    df.to_csv(DATA_PATH, index=False)
    print("Dataset generated:", DATA_PATH)

if __name__ == "__main__":
    generate_data()
