import os
import gc
import shutil
import subprocess
import cv2
import numpy as np
import streamlit as st
import yt_dlp
import imageio_ffmpeg

# ==============================================================================
# FFmpeg Path Setup
# ==============================================================================
# Prefer system ffmpeg if present; fallback to imageio-ffmpeg binary
if shutil.which("ffmpeg"):
    FFMPEG_EXE = "ffmpeg"
else:
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

ffmpeg_dir = os.path.dirname(FFMPEG_EXE) if FFMPEG_EXE != "ffmpeg" else ""
if ffmpeg_dir:
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

st.cache_data.clear()

# ==============================================================================
# UI Setup
# ==============================================================================
st.title("Universal Motion-Centered Video/GIF Maker")
st.markdown("""
Extract motion-centered clips or animated GIFs from online videos with automatic frame cropping.
""")

video_url = st.text_input("Video URL", "")
start_time = st.text_input("Start Time (HH:MM:SS)", "00:00:05")
clip_duration = st.text_input("Clip Duration (seconds)", "5")
output_format = st.selectbox("Output Format", ["mp4", "gif"])
orientation = st.selectbox("Orientation", ["horizontal", "vertical", "square"])

user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'

def parse_seconds(time_str):
    """Converts HH:MM:SS or integer strings to float seconds."""
    try:
        parts = str(time_str).split(':')
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(time_str)
    except ValueError:
        return 0.0

def process_video_pipeline(url, start, duration, output_path):
    """
    Two-stage robust pipeline:
    Stage 1: Native python HTTP/yt-dlp download to local file (No FFmpeg streaming)
    Stage 2: Standard local FFmpeg clip trim
    """
    clean_url = url.strip()
    raw_download_path = "raw_source.mp4"
    
    if os.path.exists(raw_download_path):
        try: os.remove(raw_download_path)
        except Exception: pass

    # Phase 1: Download raw source safely
    if clean_url.lower().endswith(('.mp4', '.m4v', '.mov', '.webm')):
        # Direct file download via python
        import urllib.request
        req = urllib.request.Request(clean_url, headers={'User-Agent': user_agent})
        with urllib.request.urlopen(req) as response, open(raw_download_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
    else:
        # Web URL download via yt-dlp native HTTP downloader
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best',
            'quiet': True,
            'no_warnings': True,
            'user_agent': user_agent,
            'nocheckcertificate': True,
            'outtmpl': raw_download_path,
            'force_overwrites': True,
            'hls_prefer_native': True,
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}}
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([clean_url])

    if not os.path.exists(raw_download_path) or os.path.getsize(raw_download_path) == 0:
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="Failed to download video file from host.")

    # Phase 2: Local FFmpeg trim & transcode (100% safe from HTTP network crashes)
    trim_cmd = [
        FFMPEG_EXE, '-y',
        '-threads', '1',
        '-ss', str(start),
        '-i', raw_download_path,
        '-t', str(duration),
        '-c:v', 'libx264',
        '-crf', '26',
        '-preset', 'ultrafast',
        '-c:a', 'aac',
        '-b:a', '96k',
        output_path
    ]
    
    res = subprocess.run(trim_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Cleanup raw download file to save space
    try: os.remove(raw_download_path)
    except Exception: pass

    return res

# ==============================================================================
# Main Processing Logic
# ==============================================================================
if st.button("Generate Clip"):
    if not video_url.strip():
        st.error("Please enter a valid Video URL.")
    else:
        with st.spinner("Processing clip... please wait."):
            out = f"output.{output_format}"
            tmp = "temp.mp4"

            # Cleanup previous artifacts safely
            for f in [tmp, out, "temp_cropped.mp4", "raw_source.mp4"]:
                if os.path.exists(f):
                    try: 
                        os.remove(f)
                    except Exception: 
                        pass

            # Step 1: Download & Transcode via decoupled pipeline
            try:
                r = process_video_pipeline(video_url, start_time, clip_duration, tmp)
            except Exception as ex:
                r = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=str(ex))

            # Check if output file was created successfully
            if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
                st.error("Step 1 Failed: Process could not complete.")
                st.subheader("Error Output Log")
                st.code(r.stderr if r.stderr else "Video source could not be downloaded or processed.")
            else:
                # Step 2: Motion Detection & Dynamic Cropping
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
                            if not ret: 
                                break
                            frame_count += 1
                            if frame_count % 5 != 0: 
                                continue

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
                        if target_w % 2 != 0: 
                            target_w -= 1

                        crop_tmp = "temp_cropped.mp4"
                        crop_cmd = [
                            FFMPEG_EXE, '-y',
                            '-threads', '1',
                            '-i', tmp,
                            '-vf', f"crop={target_w}:{orig_h}:{left_bound}:0",
                            '-c:a', 'copy',
                            crop_tmp
                        ]
                        crop_res = subprocess.run(crop_cmd, capture_output=True, text=True)
                        
                        if crop_res.returncode == 0 and os.path.exists(crop_tmp):
                            os.remove(tmp)
                            shutil.move(crop_tmp, tmp)
                        else:
                            st.warning("Cropping step failed, falling back to original video dimensions.")
                else:
                    cap.release()

                gc.collect()

                # Step 3: Format Rendering (MP4 vs GIF)
                if output_format == "mp4":
                    shutil.move(tmp, out)
                else:
                    try:
                        dur = parse_seconds(clip_duration)
                    except Exception:
                        dur = 5.0

                    if dur > 12:
                        target_fps, target_width = 10, (320 if orientation == "vertical" else 360)
                    elif dur > 6:
                        target_fps, target_width = 12, (360 if orientation == "vertical" else 400)
                    else:
                        target_fps, target_width = 14, (400 if orientation == "vertical" else 480)

                    fast_gif_cmd = [
                        FFMPEG_EXE, '-y',
                        '-threads', '1',
                        '-i', tmp,
                        '-vf', f"fps={target_fps},scale={target_width}:-1:flags=bilinear,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
                        out
                    ]
                    gif_res = subprocess.run(fast_gif_cmd, capture_output=True, text=True)

                    if gif_res.returncode != 0:
                        st.error("GIF conversion failed:")
                        st.code(gif_res.stderr)

                    if os.path.exists(tmp): 
                        os.remove(tmp)

            gc.collect()

            # Step 4: Preview Output
            if os.path.exists(out) and os.path.getsize(out) > 0:
                st.success("Clip generated successfully!")
                st.subheader("Preview")
                if output_format == "mp4":
                    st.video(out)
                else:
                    st.image(out)
