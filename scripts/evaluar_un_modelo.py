from ultralytics import YOLO
from tabulate import tabulate
import os
import csv

# ===========================
# CLASES
# ===========================
class_names = ["D00", "D10", "D20", "D40"]

# ===========================
# MODELOS
# ===========================
model_paths = [
    "C:\\Users\\Jorge\\Desktop\\Validacion\\models\\detect_ChinaD\\train\\weights\\last.pt",
    "C:\\Users\\Jorge\\Desktop\\Validacion\\models\\detect_ChinaM\\train\\weights\\last.pt",
    "C:\\Users\\Jorge\\Desktop\\Validacion\\models\\detect_Czech\\train\\weights\\last.pt",
    "C:\\Users\\Jorge\\Desktop\\Validacion\\models\\detect_India\\train\\weights\\last.pt",
    "C:\\Users\\Jorge\\Desktop\\Validacion\\models\\detect_Japan\\train\\weights\\last.pt",
    "C:\\Users\\Jorge\\Desktop\\Validacion\\models\\detect_Norway\\train\\weights\\last.pt",
    "C:\\Users\\Jorge\\Desktop\\Validacion\\models\\detect_UnitedStates\\train\\weights\\last.pt"
]

# ===========================
# YAML TEST FILES
# ===========================
yaml_paths = [
    "C:\\Users\\Jorge\\Desktop\\Validacion\\yaml\\yaml_test\\foldtest_all-ChinaD.yaml",
    "C:\\Users\\Jorge\\Desktop\\Validacion\\yaml\\yaml_test\\foldtest_All-ChinaM.yaml",
    "C:\\Users\\Jorge\\Desktop\\Validacion\\yaml\\yaml_test\\foldtest_All-Czech.yaml",
    "C:\\Users\\Jorge\\Desktop\\Validacion\\yaml\\yaml_test\\foldtest_All-India.yaml",
    "C:\\Users\\Jorge\\Desktop\\Validacion\\yaml\\yaml_test\\foldtest_All-Japan.yaml",
    "C:\\Users\\Jorge\\Desktop\\Validacion\\yaml\\yaml_test\\foldtest_all-Norway.yaml",
    "C:\\Users\\Jorge\\Desktop\\Validacion\\yaml\\yaml_test\\foldtest_All-UnitedS.yaml"
]

rows = []

# ===========================
# VALIDACIÓN CRUZADA
# ===========================
for model_path in model_paths:
    model_dir = os.path.dirname(os.path.dirname(os.path.dirname(model_path)))
    model_name = os.path.basename(model_dir)

    model = YOLO(model_path)

    for yaml_path in yaml_paths:
        dataset_name = os.path.splitext(os.path.basename(yaml_path))[0]

        print(f"Validando {model_name} en {dataset_name}...")

        results = model.val(data=yaml_path, split="test", imgsz=640)

        # ===========================
        # MÉTRICAS GLOBALES
        # ===========================
        mp = float(results.box.mp)
        mr = float(results.box.mr)
        map50 = float(results.box.map50)
        map5095 = float(results.box.map)

        # ===========================
        # MÉTRICAS POR CLASE
        # ===========================
        p_cls = getattr(results.box, "p", None)
        r_cls = getattr(results.box, "r", None)
        ap50_cls = getattr(results.box, "ap50", None)
        ap_cls = getattr(results.box, "ap", None)

        row = [
            model_name,
            dataset_name,
            map50,
            map5095,
            mp,
            mr
        ]

        # Añadir métricas por clase en orden D00, D10, D20, D40
        if p_cls is not None:
            row += [float(x) for x in p_cls]
        if r_cls is not None:
            row += [float(x) for x in r_cls]
        if ap50_cls is not None:
            row += [float(x) for x in ap50_cls]
        if ap_cls is not None:
            row += [float(x) for x in ap_cls]

        rows.append(row)

# ===========================
# CABECERAS CSV
# ===========================
headers = [
    "Modelo",
    "Dataset",
    "mAP50",
    "mAP50-95",
    "Precision_media",
    "Recall_media"
]

headers += [f"P_{c}" for c in class_names]
headers += [f"R_{c}" for c in class_names]
headers += [f"AP50_{c}" for c in class_names]
headers += [f"AP5095_{c}" for c in class_names]

# ===========================
# IMPRIMIR TABLA RESUMEN
# ===========================
print("\nRESULTADOS:\n")
print(tabulate(rows, headers=headers, tablefmt="grid"))

# ===========================
# GUARDAR CSV
# ===========================
csv_path = "C:\\Users\\Jorge\\Desktop\\Validacion\\resultados_validacion_por_clase.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)

print(f"\nArchivo CSV guardado en: {csv_path}\n")
