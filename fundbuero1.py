import streamlit as st
import numpy as np
import os
import json
import uuid
from datetime import datetime
from PIL import Image
from ultralytics import YOLO

# ----------------------------
# 1. KONFIGURATION
# ----------------------------
UPLOAD_DIR = "uploads"
DB_FILE = "fundstuecke.json"

# Optionale "schöne" Namen für bekannte YOLO-Objekte
PRETTY_NAMES = {
    "cell phone": "Smartphone/Elektronik",
    "laptop": "Computer",
    "backpack": "Taschen & Rucksäcke",
    "handbag": "Taschen & Rucksäcke",
    "umbrella": "Regenschirme",
    "bottle": "Trinkflaschen"
}

# Verzeichnisse initialisieren
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def load_db():
    if not os.path.exists(DB_FILE):
        return []
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

@st.cache_resource
def load_yolo():
    # Lädt das Modell (lädt beim ersten Mal automatisch die .pt Datei)
    return YOLO("yolov8n.pt")

model = load_yolo()

# ----------------------------
# 2. UI DESIGN
# ----------------------------
st.set_page_config(page_title="KI-Fundbüro", layout="wide", page_icon="🔍")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 KI-Fundbüro")
st.subheader("Automatisches Erkennen und Sortieren von Fundgegenständen")

tab1, tab2 = st.tabs(["📸 Fund melden", "📦 Inventar durchsuchen"])

# ==========================================
# TAB 1: FUND MELDEN
# ==========================================
with tab1:
    st.header("Neues Fundstück erfassen")
    uploaded_file = st.file_uploader("Bild hochladen oder Kamera nutzen", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        results = model(img)
        
        # Resultat-Bild mit Boxen erzeugen
        res_plotted = results[0].plot()
        res_image = Image.fromarray(res_plotted[:, :, ::-1])
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(res_image, caption="KI-Analyse", use_container_width=True)
        
        with col2:
            if len(results[0].boxes) > 0:
                # Das am besten erkannte Objekt
                top_box = results[0].boxes[0]
                label_en = model.names[int(top_box.cls[0])]
                conf = float(top_box.conf[0])
                
                # Dynamische Kategorie-Zuweisung
                # Wenn in PRETTY_NAMES vorhanden, nimm das, sonst den YOLO-Namen (z.B. "Vase")
                kategorie = PRETTY_NAMES.get(label_en, label_en.capitalize())
                
                st.success(f"Objekt erkannt: **{kategorie}**")
                st.write(f"Sicherheit: {conf:.1%}")
                
                if st.button(f"💾 In '{kategorie}' speichern"):
                    item_id = str(uuid.uuid4())
                    img_name = f"{item_id}.jpg"
                    img_path = os.path.join(UPLOAD_DIR, img_name)
                    
                    # Bild lokal speichern
                    res_image.save(img_path)
                    
                    # Datenbank-Eintrag
                    db = load_db()
                    db.append({
                        "id": item_id,
                        "label": label_en,
                        "category": kategorie,
                        "date": datetime.now().strftime("%d.%m.%Y, %H:%M"),
                        "img_path": img_path
                    })
                    save_db(db)
                    st.balloons()
                    st.info(f"Gespeichert unter Kategorie: {kategorie}")
            else:
                st.warning("Kein Objekt erkannt. Bitte versuche es mit einem anderen Foto.")

# ==========================================
# TAB 2: INVENTAR & GRUPPIERUNG
# ==========================================
with tab2:
    db = load_db()
    
    if not db:
        st.info("Das Inventar ist derzeit leer.")
    else:
        # Dynamische Kategorien aus der DB ziehen
        kategorien_in_db = sorted(list(set([item["category"] for item in db])))
        
        filter_auswahl = st.multiselect(
            "Filter nach Kategorien:", 
            options=kategorien_in_db, 
            default=kategorien_in_db
        )
        
        st.write("---")
        
        # Grid-Anzeige
        cols = st.columns(4)
        filtered_items = [i for i in db if i["category"] in filter_auswahl]
        
        for idx, item in enumerate(filtered_items):
            with cols[idx % 4]:
                if os.path.exists(item["img_path"]):
                    st.image(item["img_path"], use_container_width=True)
                
                st.markdown(f"**{item['category']}**")
                st.caption(f"📅 {item['date']}")
                
                if st.button("✅ Abgeholt", key=item["id"]):
                    # Aus DB entfernen
                    new_db = [x for x in db if x["id"] != item["id"]]
                    save_db(new_db)
                    # Bild löschen
                    if os.path.exists(item["img_path"]):
                        os.remove(item["img_path"])
                    st.rerun()
