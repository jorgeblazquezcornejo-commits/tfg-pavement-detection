from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import json
from urllib.parse import quote

import pandas as pd
import folium
from folium.plugins import MarkerCluster
import torch
from ultralytics import YOLO
from geopy.distance import geodesic


# =========================
# CONFIG
# =========================
@dataclass(frozen=True)
class Config:
    model_path: Path = Path(r"C:\Users\Jorge\Desktop\Validacion\models\detect_ChinaM\train\weights\last.pt")
    test_rivas_dir: Path = Path(r"C:\Users\Jorge\Desktop\Validacion\TestRivass")

    output_global_html: Path = Path(__file__).resolve().parent / "mapa_detecciones_global.html"
    output_global_csv: Path = Path(__file__).resolve().parent / "detecciones_global_nms.csv"

    # Inferencia (sube para “menos grietas”)
    conf_inferencia: float = 0.70
    image_ext: str = ".jpg"

    # NMS espacial (sube para “menos repetición”)
    dist_threshold_m: float = 30.0
    conf_threshold_nms: float = 0.70

    zoom_start: int = 16

    # Límite duro de detecciones finales globales (menos grietas)
    top_k_global: int = 500


# =========================
# CLASES
# =========================
CLASS_MAP = {
    0: ("D00", "Longitudinal Crack"),
    1: ("D10", "Transverse Crack"),
    2: ("D20", "Alligator Crack"),
    3: ("D40", "Pothole"),
}

CLASS_HEX = {
    "D00": "#d62728",
    "D10": "#1f77b4",
    "D20": "#ff7f0e",
    "D40": "#9467bd",
    "UNK": "#7f7f7f",
}


def class_info(class_id: int) -> tuple[str, str]:
    return CLASS_MAP.get(class_id, ("UNK", f"Class {class_id}"))


# =========================
# GPS CSV (tolerante + recursivo)
# =========================
def find_gps_csv(scenario_dir: Path) -> Path | None:
    """
    Busca el CSV GPS REAL del escenario. Devuelve Path si lo encuentra, o None.
    Busca también dentro de subcarpetas (rglob).
    """

    def _valid(p: Path) -> bool:
        n = p.name.lower()
        return (p.suffix.lower() == ".csv") and (not n.startswith("detecciones")) and ("detecciones" not in n)

    patterns = [
        "*frame_gps_interp*.csv",
        "*frame_gps_inter*.csv",
        "*frame_gps*.csv",
        "*gps*.csv",
    ]
    for pat in patterns:
        cand = [p for p in sorted(scenario_dir.rglob(pat)) if _valid(p)]
        if cand:
            return cand[0]

    any_csv = [p for p in sorted(scenario_dir.rglob("*.csv")) if _valid(p)]
    heuristic = [p for p in any_csv if any(k in p.name.lower() for k in ["frame", "lat", "lon", "long", "gps"])]
    if heuristic:
        return heuristic[0]

    return None


def build_paths(cfg: Config, scenario_dir: Path) -> dict:
    return {
        "scenario_dir": scenario_dir,
        "image_folder": scenario_dir / "img",
        "gps_csv": find_gps_csv(scenario_dir),  # puede ser None
        "detecciones_csv": scenario_dir / "detecciones.csv",
        "detecciones_con_gps_csv": scenario_dir / "detecciones_con_gps.csv",
        "detecciones_con_gps_nms_csv": scenario_dir / "detecciones_con_gps_nms.csv",
    }


# =========================
# INFERENCIA YOLO
# =========================
def run_inference_yolo(cfg: Config, scenario_dir: Path, image_folder: Path, detecciones_csv: Path) -> pd.DataFrame:
    model = YOLO(str(cfg.model_path))
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(cfg.image_ext)])

    detections: list[dict] = []

    for img_file in image_files:
        try:
            frame = int(Path(img_file).stem)
        except ValueError:
            continue

        img_path = image_folder / img_file
        if not img_path.exists():
            continue

        # Inferencia (silenciosa)
        try:
            results = model.predict(
                source=str(img_path),
                conf=cfg.conf_inferencia,
                save=False,
                verbose=False,
            )
        except Exception:
            continue

        if results[0].boxes is None or len(results[0].boxes) == 0:
            del results
            torch.cuda.empty_cache()
            continue

        for k, box in enumerate(results[0].boxes.data.tolist(), start=1):
            x1, y1, x2, y2, conf, cls = box
            class_code, class_name = class_info(int(cls))

            detections.append(
                {
                    "scenario": scenario_dir.name,
                    "frame": frame,
                    "image": img_file,  # nombre REAL
                    "bb_id": k,
                    "class_id": int(cls),
                    "class_code": class_code,
                    "class_name": class_name,
                    "confidence": float(conf),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }
            )

        del results
        torch.cuda.empty_cache()

    df = pd.DataFrame(detections)
    df.to_csv(detecciones_csv, index=False)
    return df


# =========================
# GPS + NMS
# =========================
def merge_with_gps(df_det: pd.DataFrame, gps_csv: Path, output_csv: Path) -> pd.DataFrame:
    df_gps = pd.read_csv(gps_csv)
    df = pd.merge(df_det, df_gps, on="frame", how="left")
    df.to_csv(output_csv, index=False)
    return df


def nms_spatial_per_class(df_class: pd.DataFrame, dist_threshold_m: float) -> pd.DataFrame:
    kept = []
    used = set()

    idxs = list(df_class.index)
    for i in idxs:
        if i in used:
            continue

        p1 = (df_class.loc[i, "lat"], df_class.loc[i, "long"])
        group = [i]

        for j in idxs:
            if j == i or j in used:
                continue
            p2 = (df_class.loc[j, "lat"], df_class.loc[j, "long"])
            if geodesic(p1, p2).meters <= dist_threshold_m:
                group.append(j)

        best = df_class.loc[group].sort_values("confidence", ascending=False).index[0]
        kept.append(best)
        used.update(group)

    return df_class.loc[kept]


def apply_nms(cfg: Config, df: pd.DataFrame, output_csv: Path) -> pd.DataFrame:
    df = df[df["confidence"] >= cfg.conf_threshold_nms].copy()
    df = df.dropna(subset=["lat", "long"])
    if df.empty:
        df.to_csv(output_csv, index=False)
        return df

    df_nms = pd.concat(
        [nms_spatial_per_class(g, cfg.dist_threshold_m) for _, g in df.groupby("class_id")],
        ignore_index=True,
    )
    df_nms.to_csv(output_csv, index=False)
    return df_nms


# =========================
# MAPA CON MODAL BB + MARKERS COLOREADOS
# (FIX OPCIÓN 2: canvas anclado al tamaño real del <img> visible)
# =========================
def create_global_map(cfg: Config, df: pd.DataFrame) -> None:
    df = df.dropna(subset=["lat", "long"]).copy()
    if df.empty:
        raise ValueError("No hay coordenadas válidas (lat/long).")

    mapa = folium.Map(location=[float(df.lat.iloc[0]), float(df.long.iloc[0])], zoom_start=cfg.zoom_start)
    cluster = MarkerCluster(name="Frames").add_to(mapa)

    modal_html = r"""
    <style>
      #bbModalBackdrop { display:none; position:fixed; top:0; left:0; right:0; bottom:0;
        background:rgba(0,0,0,0.65); z-index:9998; }
      #bbModal { display:none; position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
        width:min(92vw, 1100px); height:min(86vh, 820px); background:white; border-radius:12px; z-index:9999;
        box-shadow:0 10px 30px rgba(0,0,0,0.35); overflow:hidden; font-family:Arial, sans-serif; }
      #bbModalHeader { display:flex; align-items:center; justify-content:space-between;
        padding:10px 12px; border-bottom:1px solid #e6e6e6; }
      #bbModalTitle { font-weight:600; font-size:14px; }
      #bbModalClose { cursor:pointer; border:none; background:#f3f3f3; border-radius:8px;
        padding:6px 10px; font-size:13px; }
      #bbModalBody { display:grid; grid-template-columns: 1fr 290px; height:calc(100% - 48px); }

      #bbCanvasWrap { position:relative; background:#111; display:flex; align-items:center; justify-content:center; overflow:hidden; }
      #bbImg { max-width:100%; max-height:100%; display:block; }
      #bbCanvas { position:absolute; pointer-events:none; }

      #bbSide { border-left:1px solid #e6e6e6; padding:10px; overflow:auto; }
      .bbBtn { width:100%; text-align:left; border:1px solid #e6e6e6; background:#fff; border-radius:10px;
        padding:10px; margin-bottom:10px; cursor:pointer; }
      .bbBtn:hover { background:#fafafa; }
      .bbMeta { font-size:12px; color:#444; margin-top:6px; }
      .bbSmall { font-size:12px; color:#666; }
    </style>

    <div id="bbModalBackdrop" onclick="closeBBModal()"></div>

    <div id="bbModal">
      <div id="bbModalHeader">
        <div id="bbModalTitle"></div>
        <button id="bbModalClose" onclick="closeBBModal()">Cerrar</button>
      </div>

      <div id="bbModalBody">
        <div id="bbCanvasWrap">
          <img id="bbImg" src="" alt="imagen">
          <canvas id="bbCanvas"></canvas>
        </div>

        <div id="bbSide">
          <div class="bbSmall" id="bbInfo"></div>
          <div style="height:10px;"></div>
          <div id="bbButtons"></div>
        </div>
      </div>
    </div>

    <script>
      function closeBBModal() {
        document.getElementById("bbModalBackdrop").style.display = "none";
        document.getElementById("bbModal").style.display = "none";
        const canvas = document.getElementById("bbCanvas");
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }

      function fitCanvasToImage() {
        const img = document.getElementById("bbImg");
        const canvas = document.getElementById("bbCanvas");

        // Tamaño visible real del <img>
        const w = img.clientWidth;
        const h = img.clientHeight;

        canvas.width = w;
        canvas.height = h;

        canvas.style.width = w + "px";
        canvas.style.height = h + "px";

        // Ajustar offset del canvas porque el <img> está centrado en el wrap
        const imgRect = img.getBoundingClientRect();
        const wrapRect = img.parentElement.getBoundingClientRect();

        const left = imgRect.left - wrapRect.left;
        const top = imgRect.top - wrapRect.top;

        canvas.style.left = left + "px";
        canvas.style.top = top + "px";
      }

      function openBBModalEncoded(payloadEnc) {
        const payloadJson = decodeURIComponent(payloadEnc);
        const payload = JSON.parse(payloadJson);

        document.getElementById("bbModalBackdrop").style.display = "block";
        document.getElementById("bbModal").style.display = "block";

        document.getElementById("bbModalTitle").textContent = `Frame: ${payload.frame}`;
        document.getElementById("bbInfo").textContent =
          `GPS: ${payload.lat.toFixed(6)}, ${payload.long.toFixed(6)} | Detecciones: ${payload.bbs.length}`;

        const img = document.getElementById("bbImg");
        const canvas = document.getElementById("bbCanvas");
        const ctx = canvas.getContext("2d");

        img.onload = function() {
          fitCanvasToImage();
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        };

        img.src = payload.img_src;

        const btnWrap = document.getElementById("bbButtons");
        btnWrap.innerHTML = "";

        payload.bbs.forEach((bb) => {
          const btn = document.createElement("button");
          btn.className = "bbBtn";

          const head = document.createElement("div");
          head.textContent = `BB#${String(bb.bb_id).padStart(2,'0')}  ${bb.class_code} - ${bb.class_name}`;
          head.style.fontWeight = "600";
          head.style.fontSize = "13px";

          const meta = document.createElement("div");
          meta.className = "bbMeta";
          meta.textContent = `conf=${bb.confidence.toFixed(3)} | px=(${Math.round(bb.x1)},${Math.round(bb.y1)},${Math.round(bb.x2)},${Math.round(bb.y2)})`;

          btn.appendChild(head);
          btn.appendChild(meta);

          btn.onclick = function() { drawBB(bb); };
          btnWrap.appendChild(btn);
        });
      }

      function drawBB(bb) {
        const img = document.getElementById("bbImg");
        const canvas = document.getElementById("bbCanvas");
        const ctx = canvas.getContext("2d");

        // Recalcular por si cambia tamaño
        fitCanvasToImage();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const sx = canvas.width / img.naturalWidth;
        const sy = canvas.height / img.naturalHeight;

        const x = bb.x1 * sx;
        const y = bb.y1 * sy;
        const w = (bb.x2 - bb.x1) * sx;
        const h = (bb.y2 - bb.y1) * sy;

        const colorMap = {
          "D00": "#d62728",
          "D10": "#1f77b4",
          "D20": "#ff7f0e",
          "D40": "#9467bd",
          "UNK": "#7f7f7f"
        };
        const c = colorMap[bb.class_code] || "#7f7f7f";

        ctx.strokeStyle = c;
        ctx.lineWidth = 4;
        ctx.strokeRect(x, y, w, h);

        ctx.font = "16px Arial";
        ctx.fillStyle = c;
        const label = `BB#${String(bb.bb_id).padStart(2,'0')} ${bb.class_code} ${bb.confidence.toFixed(2)}`;
        const ty = (y - 10 > 18) ? (y - 10) : (y + 20);
        ctx.fillText(label, x, ty);
      }
    </script>
    """
    mapa.get_root().html.add_child(folium.Element(modal_html))

    root_folder_name = cfg.test_rivas_dir.name

    for (scenario, frame), g in df.groupby(["scenario", "frame"]):
        lat = float(g["lat"].iloc[0])
        lon = float(g["long"].iloc[0])

        img_filename = str(g.iloc[0]["image"])
        img_src = (Path(root_folder_name) / Path(str(scenario)) / "img" / img_filename).as_posix()

        bbs = g[["bb_id", "class_code", "class_name", "confidence", "x1", "y1", "x2", "y2"]].to_dict(orient="records")

        payload = {
            "scenario": str(scenario),
            "frame": int(frame),
            "lat": lat,
            "long": lon,
            "img_src": img_src,
            "bbs": bbs,
        }

        payload_enc = quote(json.dumps(payload, ensure_ascii=False), safe="")

        # Color marker = BB con mayor confidence
        best_row = g.loc[g["confidence"].idxmax()]
        marker_class_code = str(best_row["class_code"])
        marker_color = CLASS_HEX.get(marker_class_code, "#7f7f7f")

        popup = f"""
        <b>GPS:</b> {lat:.6f}, {lon:.6f}<br>
        <b>Detecciones:</b> {len(bbs)}<br><br>
        <button onclick="openBBModalEncoded('{payload_enc}')"
                style="padding:8px 10px;border:1px solid #ddd;border-radius:10px;background:#fff;cursor:pointer;">
          Abrir imagen
        </button>
        """

        icon = folium.DivIcon(
            html=f"""
            <div style="
                width:10px;height:10px;
                background:{marker_color};
                border:2px solid white;
                border-radius:50%;
                box-shadow:0 0 2px rgba(0,0,0,0.4);
            "></div>
            """
        )

        folium.Marker(
            location=[lat, lon],
            icon=icon,
            popup=folium.Popup(popup, max_width=280),
        ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(mapa)
    mapa.save(str(cfg.output_global_html))


# =========================
# MAIN
# =========================
def main() -> None:
    cfg = Config()

    if not cfg.test_rivas_dir.exists():
        raise FileNotFoundError(f"No existe test_rivas_dir: {cfg.test_rivas_dir}")
    if not cfg.model_path.exists():
        raise FileNotFoundError(f"No existe model_path: {cfg.model_path}")

    scenarios = sorted([d for d in cfg.test_rivas_dir.iterdir() if d.is_dir()])
    global_parts: list[pd.DataFrame] = []

    for scenario_dir in scenarios:
        try:
            p = build_paths(cfg, scenario_dir)
            if not p["image_folder"].exists():
                continue

            if p["gps_csv"] is None:
                print(f"[{scenario_dir.name}] Sin GPS CSV -> se omite este escenario.")
                continue

            df_det = run_inference_yolo(cfg, scenario_dir, p["image_folder"], p["detecciones_csv"])
            df_gps = merge_with_gps(df_det, p["gps_csv"], p["detecciones_con_gps_csv"])
            df_nms = apply_nms(cfg, df_gps, p["detecciones_con_gps_nms_csv"])

            if not df_nms.empty:
                global_parts.append(df_nms)

        except Exception as e:
            print(f"[{scenario_dir.name}] Error: {e}")

    if not global_parts:
        raise RuntimeError("No hay detecciones válidas tras GPS+NMS.")

    df_global = pd.concat(global_parts, ignore_index=True)

    # Limitar a Top-K global (menos grietas)
    df_global = df_global.sort_values("confidence", ascending=False).head(cfg.top_k_global)

    df_global.to_csv(cfg.output_global_csv, index=False)

    create_global_map(cfg, df_global)


if __name__ == "__main__":
    main()
