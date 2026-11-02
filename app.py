
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os

st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI-Based Object Detection and Tracking")
st.write("Detect and track objects using YOLO and Computer Vision.")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

uploaded_file = st.file_uploader(
    "Upload an Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

mode = st.radio(
    "Choose Detection Mode",
    ["Object Detection", "Object Tracking"],
    horizontal=True
)

if uploaded_file:
    model = load_model()
    ext = os.path.splitext(uploaded_file.name)[1].lower()

    if ext in [".jpg", ".jpeg", ".png"]:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width="stretch")

        if st.button("Start Detection"):
            results = model.predict(image, conf=0.25)
            annotated = results[0].plot()
            annotated = cv2.cvtColor(
                annotated, cv2.COLOR_BGR2RGB
            )

            st.subheader("Detection Result")
            st.image(annotated, width="stretch")

            names = results[0].names
            counts = {}
            for cls in results[0].boxes.cls.tolist():
                name = names[int(cls)]
                counts[name] = counts.get(name, 0) + 1

            st.subheader("Detected Objects")
            st.write(counts)
            st.success(f"Total objects detected: {sum(counts.values())}")

    else:
        st.video(uploaded_file)

        if st.button("Start Video Processing"):
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=ext
            ) as temp:
                temp.write(uploaded_file.read())
                video_path = temp.name

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25

            output_path = tempfile.mktemp(suffix=".mp4")
            writer = None
            progress = st.progress(0)
            display = st.empty()

            frame_count = 0
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if mode == "Object Tracking":
                        result = model.track(
                            frame, persist=True, conf=0.25,
                            verbose=False
                        )[0]
                    else:
                        result = model.predict(
                            frame, conf=0.25, verbose=False
                        )[0]

                    annotated = result.plot()

                    if writer is None:
                        height, width = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            output_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps, (width, height)
                        )

                    writer.write(annotated)
                    frame_count += 1

                    if frame_count % 5 == 0:
                        display.image(
                            cv2.cvtColor(
                                annotated, cv2.COLOR_BGR2RGB
                            ),
                            channels="RGB",
                            width="stretch"
                        )

                    if total_frames:
                        progress.progress(
                            min(frame_count / total_frames, 1.0)
                        )
            finally:
                cap.release()
                if writer:
                    writer.release()
                if os.path.exists(video_path):
                    os.remove(video_path)

            if os.path.exists(output_path) and os.path.getsize(output_path):
                st.success("Video processing completed!")
                with open(output_path, "rb") as file:
                    st.download_button(
                        "Download Processed Video",
                        data=file,
                        file_name="detected_video.mp4",
                        mime="video/mp4"
                    )
                os.remove(output_path)
            else:
                st.error("Video processing failed. Try another video.")

else:
    st.info("Upload an image or video to begin.")
