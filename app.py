import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import base64
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
st.set_page_config(page_title='Name of This Butterflies', page_icon='butterfly')

model = tf.keras.models.load_model("saved_model/butterflies.hdf5")
### load file
st.markdown("<h1 style='text-align: center;'>ผีเสื้อตัวนี้ชื่ออะไร เดี๋ยวตอบให้!</h1>", unsafe_allow_html=True)

file_ = open("img/is-this-a-pigeon-butterfly.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<p style="text-align:center;"><img src="data:image/gif;base64,{data_url}" alt="butterfly gif"></p>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("อัปโหลดรูปภาพเพื่อค้นหา", type="jpg")
st.write("รองรับมากกว่า 75 สายพันธุ์ โดยสามารถดูรายชื่อได้ [ที่นี่](https://github.com/0045w/BSIC/blob/main/class_dict.csv)")

map_dict = {0: 'ADONIS', 
            1: 'AFRICAN GIANT SWALLOWTAIL', 
            2: 'AMERICAN SNOOT', 
            3: 'AN 88', 
            4: 'APPOLLO', 
            5: 'ATALA', 
            6: 'BANDED ORANGE HELICONIAN', 
            7: 'BANDED PEACOCK', 
            8: 'BECKERS WHITE', 
            9: 'BLACK HAIRSTREAK', 
            10: 'BLUE MORPHO', 
            11: 'BLUE SPOTTED CROW', 
            12: 'BROWN SIPROETA', 
            13: 'CABBAGE WHITE', 
            14: 'CAIRNS BIRDWING', 
            15: 'CHECQUERED SKIPPER', 
            16: 'CHESTNUT', 
            17: 'CLEOPATRA', 
            18: 'CLODIUS PARNASSIAN', 
            19: 'CLOUDED SULPHUR', 
            20: 'COMMON BANDED AWL', 
            21: 'COMMON WOOD-NYMPH', 
            22: 'COPPER TAIL', 
            23: 'CRECENT', 
            24: 'CRIMSON PATCH', 
            25: 'DANAID EGGFLY', 
            26: 'EASTERN COMA', 
            27: 'EASTERN DAPPLE WHITE', 
            28: 'EASTERN PINE ELFIN', 
            29: 'ELBOWED PIERROT', 
            30: 'GOLD BANDED', 
            31: 'GREAT EGGFLY', 
            32: 'GREAT JAY', 
            33: 'GREEN CELLED CATTLEHEART', 
            34: 'GREY HAIRSTREAK', 
            35: 'INDRA SWALLOW', 
            36: 'IPHICLUS SISTER', 
            37: 'JULIA', 
            38: 'LARGE MARBLE', 
            39: 'MALACHITE', 
            40: 'MANGROVE SKIPPER', 
            41: 'MESTRA', 
            42: 'METALMARK', 
            43: 'MILBERTS TORTOISESHELL', 
            44: 'MONARCH', 
            45: 'MOURNING CLOAK', 
            46: 'ORANGE OAKLEAF', 
            47: 'ORANGE TIP', 
            48: 'ORCHARD SWALLOW', 
            49: 'PAINTED LADY', 
            50: 'PAPER KITE', 
            51: 'PEACOCK', 
            52: 'PINE WHITE', 
            53: 'PIPEVINE SWALLOW', 
            54: 'POPINJAY', 
            55: 'PURPLE HAIRSTREAK', 
            56: 'PURPLISH COPPER', 
            57: 'QUESTION MARK', 
            58: 'RED ADMIRAL', 
            59: 'RED CRACKER', 
            60: 'RED POSTMAN', 
            61: 'RED SPOTTED PURPLE', 
            62: 'SCARCE SWALLOW', 
            63: 'SILVER SPOT SKIPPER', 
            64: 'SLEEPY ORANGE', 
            65: 'SOOTYWING', 
            66: 'SOUTHERN DOGFACE', 
            67: 'STRAITED QUEEN', 
            68: 'TROPICAL LEAFWING', 
            69: 'TWO BARRED FLASHER', 
            70: 'ULYSES', 
            71: 'VICEROY', 
            72: 'WOOD SATYR', 
            73: 'YELLOW SWALLOW TAIL', 
            74: 'ZEBRA LONG WING'}

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("ค้นหาสายพันธุ์")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.subheader("ผีเสื้อตัวนี้สายพันธุ์ {}".format(map_dict [prediction]))



