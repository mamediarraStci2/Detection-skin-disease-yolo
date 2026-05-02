import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import time
import pandas as pd
import colorsys
import os

st.set_page_config(page_title='Skin Disease Detector', page_icon='🔬', layout='wide')

st.markdown('<h1 style="text-align:center;color:#90caf9;font-size:2.2rem">🔬 Détection de Maladies de Peau</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#78909c">Modèle YOLOv8-m — Fine-tuné sur Skin Disease Dataset</p>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = 'best.pt'
    # Try local file first, otherwise use YOLOv8m default
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        with st.spinner('📥 Téléchargement du modèle YOLOv8m...'):
            return YOLO('yolov8m.pt')

model   = load_model()
CLASSES = model.names

def get_color(i):
    r, g, b = colorsys.hsv_to_rgb((i * 30 % 180) / 180, 0.85, 0.95)
    return (int(r*255), int(g*255), int(b*255))

with st.sidebar:
    st.header('⚙️ Paramètres')
    conf = st.slider('Seuil de confiance', 0.05, 0.95, 0.25, 0.05)
    iou  = st.slider('Seuil IoU',          0.10, 0.90, 0.45, 0.05)
    show = st.checkbox('Afficher les scores', value=True)
    st.divider()
    st.markdown('**Classes détectées :**')
    for i, n in CLASSES.items():
        st.markdown(f'- `{n}`')

up = st.file_uploader('📤 Téléverse une image de lésion cutanée', type=['jpg','jpeg','png'])

if up:
    try:
        pil = Image.open(up).convert('RGB')
    except Exception:
        st.warning('⚠️ Image corrompue.')
        st.stop()

    with st.spinner('🔬 Analyse en cours...'):
        arr  = np.array(pil)
        bgr  = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        t0   = time.time()
        res  = model.predict(bgr, conf=conf, iou=iou, verbose=False)
        ms   = (time.time() - t0) * 1000
        out  = arr.copy()
        dets = []
        for box in res[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cid = int(box.cls[0])
            cf  = float(box.conf[0])
            col = get_color(cid)
            cv2.rectangle(out, (x1,y1), (x2,y2), col, 3)
            lbl = f'{CLASSES[cid]} {cf:.2f}' if show else CLASSES[cid]
            cv2.putText(out, lbl, (x1, max(y1-8,12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            dets.append({'Classe': CLASSES[cid], 'Confiance': round(cf,3),
                         'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    n   = len(dets)
    avg = round(sum(d['Confiance'] for d in dets) / max(n,1), 3)
    dom = max(set(d['Classe'] for d in dets), key=lambda c: sum(1 for d in dets if d['Classe']==c)) if dets else '—'

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('🎯 Détections',  str(n))
    c2.metric('📌 Dominante',   dom)
    c3.metric('📊 Confiance',   str(avg))
    c4.metric('⚡ Temps',       f'{ms:.0f} ms')

    col1, col2 = st.columns(2)
    col1.image(pil,                  caption='Image originale',   width=600)
    col2.image(Image.fromarray(out), caption='Détections YOLOv8', width=600)

    if dets:
        st.subheader('📋 Détail des détections')
        st.dataframe(pd.DataFrame(dets), use_container_width=True)
    else:
        st.info('ℹ️ Aucune détection — réduis le seuil de confiance.')

    if st.button('🔄 Tester une nouvelle image'):
        st.rerun()

else:
    st.info('👆 Téléverse une image JPG/PNG pour lancer la détection.')
    st.markdown('### 💡 Comment ça marche ?')
    st.markdown('1. Téléverse une image de lésion cutanée')
    st.markdown('2. YOLOv8-m détecte et localise les maladies automatiquement')
    st.markdown('3. Les boîtes s\'affichent avec le nom et le score de confiance')