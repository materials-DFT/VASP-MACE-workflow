import os
import re

def modify_incar(incar_path):
    with open(incar_path, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    flags_found = {
        "ISTART": False,
        "ICHARG": False,
        "IBRION": False,
        "LREAL": False,
        "NEDOS": False,
        "EMIN": False,
        "EMAX": False
    }

    for line in lines:
        original = line.strip()
        key = re.sub(r'^#+', '', original).split('=')[0].strip() if '=' in original else None

        if key == "ISTART":
            updated_lines.append("ISTART = 1\n")
            flags_found["ISTART"] = True
        elif key == "ICHARG":
            updated_lines.append("ICHARG = 1\n")
            flags_found["ICHARG"] = True
        elif key == "IBRION":
            updated_lines.append("IBRION = -1\n")
            flags_found["IBRION"] = True
        elif key == "LREAL":
            updated_lines.append("LREAL = .FALSE.\n")
            flags_found["LREAL"] = True
        elif key == "NEDOS":
            updated_lines.append("NEDOS = 1501\n")
            flags_found["NEDOS"] = True
        elif key == "EMIN":
            updated_lines.append("EMIN = -20.0\n")
            flags_found["EMIN"] = True
        elif key == "EMAX":
            updated_lines.append("EMAX = 10.0\n")
            flags_found["EMAX"] = True
        elif key in {"ISIF", "POTIM", "NSW"}:
            # comment out even if already commented
            updated_lines.append(f"# {original}\n")
        else:
            updated_lines.append(line)

    # Add missing parameters
    if not flags_found["ISTART"]:
        updated_lines.append("ISTART = 1\n")
    if not flags_found["ICHARG"]:
        updated_lines.append("ICHARG = 1\n")
    if not flags_found["IBRION"]:
        updated_lines.append("IBRION = -1\n")
    if not flags_found["LREAL"]:
        updated_lines.append("LREAL = .FALSE.\n")
    if not flags_found["NEDOS"]:
        updated_lines.append("NEDOS = 1501\n")
    if not flags_found["EMIN"]:
        updated_lines.append("EMIN = -20.0\n")
    if not flags_found["EMAX"]:
        updated_lines.append("EMAX = 10.0\n")

    # Write back the modified INCAR
    with open(incar_path, 'w') as f:
        f.writelines(updated_lines)

    print(f"✔ Updated: {incar_path}")

def main():
    cwd = os.getcwd()
    for root, dirs, files in os.walk(cwd):
        for file in files:
            if file == "INCAR":
                incar_path = os.path.join(root, file)
                modify_incar(incar_path)

if __name__ == "__main__":
    main()
