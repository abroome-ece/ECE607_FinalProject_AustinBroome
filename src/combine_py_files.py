import os

output_file = "all_python_code.txt"

with open(output_file, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py") and file != os.path.basename(__file__):
                filepath = os.path.join(root, file)

                outfile.write(f"\n{'=' * 80}\n")
                outfile.write(f"FILE: {filepath}\n")
                outfile.write(f"{'=' * 80}\n\n")

                try:
                    with open(filepath, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n\n")
                except Exception as e:
                    outfile.write(f"ERROR READING FILE: {e}\n\n")

print(f"Combined Python files written to {output_file}")