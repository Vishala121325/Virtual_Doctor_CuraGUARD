import pandas as pd
import json
import os

# Dynamically get paths
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, 'data', 'cleaned_symptom_dataset.csv')
output_path = os.path.join(base_path, 'data', 'disease_info.json')

print("🔍 Looking for file at:", csv_path)

# Load the CSV with proper encoding
try:
    df = pd.read_csv(csv_path, encoding="cp1252")
    df.columns = df.columns.str.strip()  # Clean headers
    print("✅ CSV loaded successfully!")
except FileNotFoundError:
    print(f"❌ File not found at {csv_path}")
    exit()
except Exception as e:
    print("❌ Failed to load CSV:", e)
    exit()

# Show columns for quick check
print("🧾 Columns in file:", df.columns.tolist())

# Fill missing values
df = df.fillna("")

# Storage for disease info
disease_info = {}

# Parse rows
for i, row in df.iterrows():
    try:
        disease = row["Disease"].strip().title()
        print(f"🛠️ Processing row {i}: {disease}")

        disease_info[disease] = {
            "severity": row.get("Severity Level", "").strip().title(),
            "tests": [t.strip() for t in str(row.get("Recommended Tests", "")).split(",") if t.strip()],
            "diet": row.get("Diet", "").strip(),
            "lifestyle": row.get("lifestyle", "").strip(),
            "home_remedies": [r.strip() for r in str(row.get("remedies", "")).split(",") if r.strip()],
            "medicines": [m.strip() for m in str(row.get("medicines", "")).split(",") if m.strip()]
        }

    except Exception as e:
        print(f"⚠️ Skipping row {i} due to error: {e}")

# Save JSON
if disease_info:
    try:
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(disease_info, f, indent=4)
        print(f"✅ JSON saved to: {output_path}")
    except Exception as e:
        print("❌ Failed to save JSON:", e)
else:
    print("⚠️ No disease data processed.")
