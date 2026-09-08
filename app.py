import os
import imageio_ffmpeg

# Automatically locate the static ffmpeg binary and set it in the PATH
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
import os
import gc
import subprocess
import cv2
import numpy as np
import streamlit as st
import yt_dlp

# Force cache clearance on every run to keep memory footprint low
st.cache_data.clear()

st.title("Universal Motion-Centered Video/GIF Maker")
st.markdown("""
Extract motion-centered clips or animated GIFs from online videos with automatic frame cropping. 

*If processing fails or times out, try using a shorter clip duration or a different video site.*
""")

# User Inputs (No "Save File As" or description inputs)
video_url = st.text_input("Video URL", "")
start_time = st.text_input("Start Time (HH:MM:SS)", "00:02:34")
clip_duration = st.text_input("Clip Duration (seconds)", "16")
output_format = st.selectbox("Output Format", ["mp4", "gif"])
orientation = st.selectbox("Orientation", ["horizontal", "vertical", "square"])

user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'

def get_direct_stream_url(source_url):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best',
        'quiet': True,
        'no_warnings': True,
        'user_agent': user_agent,
        'nocheckcertificate': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source_url, download=False)
            if 'url' in info:
                return info['url']
            elif 'requested_formats' in info:
                return info['requested_formats'][0]['url']
    except Exception:
        pass
    return source_url

if st.button("Generate Clip"):
    if not video_url.strip():
        st.error("Please enter a valid Video URL.")
    else:
        with st.spinner("Processing clip... please wait."):
            out = f"output.{output_format}"
            tmp = "temp.mp4"

            # Clean pre-existing temporary files
            for f in [tmp, out, "temp_cropped.mp4"]:
                if os.path.exists(f):
                    try: os.remove(f)
                    except Exception: pass

            stream_url = get_direct_stream_url(video_url)
            headers = f"User-Agent: {user_agent}\r\nReferer: {video_url}\r\n"
            
            cmd1 = [
                'ffmpeg', '-y', '-headers', headers, 
                '-ss', start_time, '-i', stream_url, 
                '-t', clip_duration, '-c:v', 'libx264', 
                '-crf', '26', '-preset', 'ultrafast', 
                '-c:a', 'aac', '-b:a', '96k', '-vsync', 'vfr', tmp
            ]
            
            r = subprocess.run(cmd1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                cap = cv2.VideoCapture(tmp)
                orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if orientation != "horizontal" and orig_h > 0:
                    scale = 240.0 / orig_h if orig_h > 240 else 1.0
                    tw, th = int(orig_w * scale), int(orig_h * scale)

                    ret, prev_frame = cap.read()
                    if ret:
                        prev_gray = cv2.cvtColor(cv2.resize(prev_frame, (tw, th)), cv2.COLOR_BGR2GRAY)
                        active_x_positions = []
                        frame_count = 0

                        while True:
                            ret, frame = cap.read()
                            if not ret: break
                            frame_count += 1
                            if frame_count % 5 != 0: continue

                            gray = cv2.cvtColor(cv2.resize(frame, (tw, th)), cv2.COLOR_BGR2GRAY)
                            delta = cv2.absdiff(prev_gray, gray)
                            _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)

                            y_indices, x_indices = np.where(thresh > 0)
                            if len(x_indices) > 0:
                                active_x_positions.extend((x_indices / scale).astype(int))
                            prev_gray = gray
                        cap.release()

                        target_w = int(orig_h * (3 / 5)) if orientation == "vertical" else int(orig_h * (4 / 5))

                        if active_x_positions:
                            center_x = int(np.median(active_x_positions))
                            left_bound = center_x - (target_w // 2)
                        else:
                            left_bound = (orig_w - target_w) // 2

                        left_bound = max(0, min(left_bound, orig_w - target_w))
                        if left_bound + target_w > orig_w:
                            target_w = orig_w - left_bound
                        if target_w % 2 != 0: target_w -= 1

                        crop_tmp = "temp_cropped.mp4"
                        subprocess.run(
                            ['ffmpeg', '-y', '-i', tmp, '-vf', f"crop={target_w}:{orig_h}:{left_bound}:0", '-c:a', 'copy', crop_tmp], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                        os.remove(tmp)
                        os.rename(crop_tmp, tmp)
                else:
                    cap.release()

                gc.collect()

                if output_format == "mp4":
                    os.rename(tmp, out)
                else:
                    try:
                        dur = float(clip_duration)
                    except ValueError:
                        dur = 16.0

                    if dur > 12:
                        target_fps = 10
                        target_width = 320 if orientation == "vertical" else 360
                    elif dur > 6:
                        target_fps = 12
                        target_width = 360 if orientation == "vertical" else 400
                    else:
                        target_fps = 14
                        target_width = 400 if orientation == "vertical" else 480

                    fast_gif_cmd = [
                        'ffmpeg', '-y', '-i', tmp,
                        '-vf', f"fps={target_fps},scale={target_width}:-1:flags=bilinear,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
                        out
                    ]
                    subprocess.run(fast_gif_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    if os.path.exists(tmp): 
                        os.remove(tmp)

            gc.collect()

            if os.path.exists(out) and os.path.getsize(out) > 0:
                st.success("File generated successfully!")
                
                st.subheader("Preview")
                if output_format == "mp4":
                    st.video(out)
                else:
                    st.image(out)
            else:
                st.error("Processing failed. Please verify the URL or try a different video site / shorter clip duration.")
