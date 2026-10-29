import json
import re
from pathlib import Path

items = [
    line.rstrip("\n").split("\t")
    for line in Path("configs/congreso_202_items.tsv").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
FEM = {
    "Muñoz de la Iglesia, Ester",
    "Belarra Urteaga, Ione",
    "Guardiola Salmerón, Mirian",
    "Díaz Pérez, Yolanda",
    "Álvarez de Toledo Peralta-Ramos, Cayetana",
    "Rodríguez de Millán Parro, María José",
    "Rodríguez García, Isabel",
    "Madrenas i Mir, Marta",
    "Vaquero Montero, Maribel",
    "Rodríguez Calleja, Patricia",
    "Redondo García, Ana María del Carmen",
    "Franco González, Silvia",
    "Saiz Delgado, Elma",
    "Jordà i Roura, Teresa",
}
blocks = []
cur = None
for i, lab in items:
    if lab.startswith("PREGUNTA"):
        cur = {"block_id": int(i), "title": lab, "interventions": []}
        blocks.append(cur)
    else:
        name = re.split(r" \(", lab)[0].strip()
        cur["interventions"].append({"id": int(i), "name": name, "label": lab})
for b in blocks:
    q = b["interventions"][0]["name"]
    assert ("Diputada Dª" in b["title"]) == (q in FEM), b["title"]
    for iv in b["interventions"]:
        iv["gender"] = "M" if iv["name"] in FEM else "H"
        if "Ministra" in iv["label"] or "Vicepresidenta" in iv["label"]:
            assert iv["gender"] == "M", iv
        if re.search(r"\(Ministro|Vicepresidente Primero", iv["label"]):
            assert iv["gender"] == "H", iv
for n, b in enumerate(blocks, 1):
    b["episode"] = f"es_congreso_202_q{n:02d}"
man = {
    "source": (
        "Congreso de los Diputados, Sesión Plenaria nº 202 (XV Legislatura), 30/09/2026, sesión de control. "
        "Vídeos oficiales de descarga del archivo audiovisual."
    ),
    "page": "https://app.congreso.es/AudiovisualCongreso/audiovisualdetalledisponible?codOrgano=400&codSesion=202&idLegislaturaElegida=15&fechaSesion=30/09/2026",
    "url_template": "https://static.congreso.es/audiovisual/video/leg15/400/15_000400_202/cortes/15_000400_202_0_27702_{id}.mp4",
    "gender_source": (
        "Tratamiento oficial publicado por el Congreso (Diputada Dª / Diputado D., "
        "Ministra / Ministro, Vicepresidenta / Vicepresidente)."
    ),
    "blocks": blocks,
}
Path("configs/congreso_202.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print("MANIFEST", len(blocks), sum(len(b["interventions"]) for b in blocks))
