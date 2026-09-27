import streamlit as st
import requests
import cv2

st.header("FACEMASK DETECTOR")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg","jpeg","png"]
)

#IMAGE PRED
if uploaded_file is not None:

    st.image(uploaded_file) #Displays the image

    files = {
        "file": uploaded_file.getvalue()
    }

    response = requests.post(
        "http://127.0.0.1:8000/image_predict",
        files=files
    )

    result = response.json()

    st.success(result["prediction"])

    st.write(result)

if st.button("GO LIVE"):

    frame_placeholder = st.empty()

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Convert frame to JPEG
        _, buffer = cv2.imencode(".jpg", frame)

        files = {
            "file": (
                "frame.jpg",
                buffer.tobytes(),
                "image/jpeg"
            )
        }

        response = requests.post(
            "http://127.0.0.1:8000/vid_predict",
            files=files
        )

        if response.status_code != 200:
            st.error(response.text)
            break

        result = response.json()

        # Draw rectangle if face found
        if "bbox" in result:

            x, y, w, h = result["bbox"]

            color = (0,255,0)

            if result["prediction"] == "MASK OFF":
                color = (0,0,255)

            cv2.rectangle(
                frame,
                (x,y),
                (x+w,y+h),
                color,
                2
            )

            cv2.putText(
                frame,
                result["prediction"],
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        frame_placeholder.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB"
        )

    cap.release()