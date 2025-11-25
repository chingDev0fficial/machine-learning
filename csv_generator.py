import os
import csv

dataset_path = "C:\\Users\\princ\\Documents\projects\\asrs\\asrs_project\\asrs_backend\\media\\uploads\\feed_users_signatures\\signatures\\validation"
output_csv = "pairs.csv"

rows = []

for folder in sorted(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    # Separate genuine vs forged
    if folder.endswith("_forge"):
        base_id = folder.replace("_forge", "")
        forged_images = sorted(os.listdir(folder_path))
        genuine_folder = os.path.join(dataset_path, base_id)
        if not os.path.exists(genuine_folder):
            continue
        genuine_images = sorted(os.listdir(genuine_folder))
    else:
        base_id = folder
        genuine_folder = folder_path
        genuine_images = sorted(os.listdir(genuine_folder))

        forge_folder = os.path.join(dataset_path, base_id + "_forge")
        forged_images = sorted(os.listdir(forge_folder)) if os.path.exists(forge_folder) else []

    # --- Genuine-Genuine pairs (label 1)
    for i in range(len(genuine_images)):
        for j in range(len(genuine_images)):
            if i == j:  
                continue  # skip identical images

            img1 = f"{base_id}/{genuine_images[i]}"
            img2 = f"{base_id}/{genuine_images[j]}"
            rows.append([img1, img2, 1])

    # --- Genuine-Forged pairs (label 0)
    for g in genuine_images:
        for f in forged_images:
            img1 = f"{base_id}/{g}"
            img2 = f"{base_id}_forge/{f}"
            rows.append([img1, img2, 0])


# Save CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["img1", "img2", "label"])
    writer.writerows(rows)

print(f"CSV generated: {output_csv}")
