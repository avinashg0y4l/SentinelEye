from zipfile import ZipFile

zip_path = "data/raw/DFIRE/D-Fire.zip"

with ZipFile(zip_path, "r") as z:
    labels = [
        name
        for name in z.namelist()
        if name.startswith("train/labels/")
        and name.endswith(".txt")
    ]

    print("Total label files:", len(labels))

    count = 0

    for label_path in labels:
        content = z.read(label_path).decode("utf-8", errors="ignore").strip()

        if content:
            print("\nFirst non-empty label:")
            print("File:", label_path)
            print("Content:")
            print(content[:500])

            # Corresponding image
            image_path = label_path.replace(
                "train/labels/",
                "train/images/"
            ).replace(".txt", ".jpg")

            print("Image:", image_path)

            count += 1

            if count >= 5:
                break