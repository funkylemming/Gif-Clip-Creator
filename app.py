import os
import subprocess
import cv2
import numpy as np
import streamlit as st
import yt_dlp
import base64
import streamlit.components.v1 as components

# Title
st.title("Universal Motion-Centered Video/GIF Maker")

# Description / Instructions
st.markdown("""
Extract clips from video URLs, automatically crop them using motion tracking, and generate optimized MP4s or GIFs under 10MB.

**How to use:**
1. Paste the direct video URL below.
2. Set your start time and clip duration.
3. Choose your preferred framing orientation and output format.
4. Click **Generate Clip** to build and download your file.
""")

# Website compatibility notice
st.warning("Note: Due to variable site structures and streaming protections, this tool may not work with every website. If a link fails, try a video from a different site or source.")

st.divider()

# User Inputs (empty defaults for URL and Save File As)
video_url = st.text_input("Video URL", "")
start_time = st.text_input("Start Time (HH:MM:SS)", "00:02:34")
clip_duration = st.text_input("Clip Duration (seconds)", "16")
save_file_as = st.text_input("Save File As", "")
output_format = st.selectbox("Output Format", ["mp4", "gif"])
orientation = st.selectbox("Orientation", ["horizontal", "vertical", "square"])

user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'

def get_direct_stream_url(source_url):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
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
    elif not save_file_as.strip():
        st.error("Please enter a name in 'Save File As'.")
    else:
        with st.spinner("Processing clip... please wait."):
            name = save_file_as.strip().replace('.mp4','').replace('.gif','')
            out = f"{name}.{output_format}"
            tmp = "temp.mp4"

            stream_url = get_direct_stream_url(video_url)
            headers = f"User-Agent: {user_agent}\r\nReferer: {video_url}\r\n"
            cmd1 = ['ffmpeg', '-y', '-headers', headers, '-ss', start_time, '-i', stream_url, '-t', clip_duration, '-c:v', 'libx264', '-crf', '22', '-preset', 'veryfast', '-c:a', 'aac', '-b:a', '128k', '-vsync', 'vfr', tmp]
            
            r = subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                cap = cv2.VideoCapture(tmp)
                orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if orientation != "horizontal":
                    scale = 360.0 / orig_h if orig_h > 360 else 1.0
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
                            if frame_count % 3 != 0: continue

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
                        subprocess.run(['ffmpeg', '-y', '-i', tmp, '-vf', f"crop={target_w}:{orig_h}:{left_bound}:0", '-c:a', 'copy', crop_tmp], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        os.remove(tmp)
                        os.rename(crop_tmp, tmp)
                else:
                    cap.release()

                if output_format == "mp4":
                    os.rename(tmp, out)
                else:
                    widths = [400, 360, 320, 280, 240] if orientation == "vertical" else ([540, 480, 420, 360, 300] if orientation == "square" else [640, 560, 480, 400, 320])
                    quality_tiers = [
                        {"fps": 18, "w_idx": 0, "colors": 256, "dither": "bayer:bayer_scale=3", "label": "Ultra (18 FPS)"},
                        {"fps": 16, "w_idx": 1, "colors": 256, "dither": "bayer:bayer_scale=3", "label": "High (16 FPS)"},
                        {"fps": 15, "w_idx": 2, "colors": 224, "dither": "bayer:bayer_scale=4", "label": "Balanced (15 FPS)"},
                        {"fps": 12, "w_idx": 3, "colors": 192, "dither": "bayer:bayer_scale=4", "label": "Compact (12 FPS)"},
                        {"fps": 10, "w_idx": 4, "colors": 160, "dither": "bayer:bayer_scale=5", "label": "Low (10 FPS)"}
                    ]

                    target_bytes = 10 * 1024 * 1024
                    sweet_spot_bytes = 8 * 1024 * 1024
                    low, high = 0, len(quality_tiers) - 1
                    best_valid_file = None

                    while low <= high:
                        mid = (low + high) // 2
                        tier = quality_tiers[mid]
                        gif_w = widths[tier["w_idx"]]
                        test_out = f"test_{mid}.gif"

                        scale_filter = f"fps={tier['fps']},scale={gif_w}:-1:flags=lanczos"
                        palette_filter = f"{scale_filter},palettegen=max_colors={tier['colors']}:stats_mode=diff"
                        render_filter = f"{scale_filter}[x];[x][1:v]paletteuse=dither={tier['dither']}:diff_mode=rectangle"

                        subprocess.run(['ffmpeg', '-y', '-i', tmp, '-vf', palette_filter, 'p.png'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        subprocess.run(['ffmpeg', '-y', '-i', tmp, '-i', 'p.png', '-filter_complex', render_filter, test_out], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                        if os.path.exists('p.png'): os.remove('p.png')

                        if os.path.exists(test_out):
                            size = os.path.getsize(test_out)
                            if size <= target_bytes:
                                if best_valid_file and os.path.exists(best_valid_file):
                                    os.remove(best_valid_file)
                                best_valid_file = test_out
                                if size >= sweet_spot_bytes: break
                                else: high = mid - 1
                            else:
                                os.remove(test_out)
                                low = mid + 1

                    if best_valid_file and os.path.exists(best_valid_file):
                        if os.path.exists(out): os.remove(out)
                        os.rename(best_valid_file, out)

                    if os.path.exists(tmp): os.remove(tmp)

            if os.path.exists(out) and os.path.getsize(out) > 0:
                st.success("File generated successfully!")
                
                with open(out, "rb") as file:
                    file_bytes = file.read()

                # Standard Download Button
                st.download_button(
                    label=f"Download {output_format.upper()}",
                    data=file_bytes,
                    file_name=out,
                    mime="video/mp4" if output_format == "mp4" else "image/gif"
                )

                # Clipboard Copy Option for GIFs
                if output_format == "gif":
                    b64_gif = base64.b64encode(file_bytes).decode("utf-8")
                    copy_html = f"""
                    <button id="copyBtn" style="
                        background-color: #ff4b4b;
                        color: white;
                        border: none;
                        padding: 0.5rem 1rem;
                        border-radius: 0.5rem;
                        cursor: pointer;
                        font-size: 14px;
                        font-weight: 500;
                        margin-top: 5px;">
                        📋 Copy GIF to Clipboard
                    </button>
                    <span id="status" style="margin-left: 10px; font-family: sans-serif; font-size: 14px; color: #333;"></span>

                    <script>
                    document.getElementById('copyBtn').addEventListener('click', async () => {{
                        const status = document.getElementById('status');
                        status.innerText = 'Copying...';
                        try {{
                            const b64Data = '{b64_gif}';
                            const res = await fetch(`data:image/gif;base64,${{b64Data}}`);
                            const blob = await res.blob();
                            
                            await navigator.clipboard.write([
                                new ClipboardItem({{ [blob.type]: blob }})
                            ]);
                            status.innerText = '✅ Copied!';
                        }} catch (err) {{
                            status.innerText = '❌ Failed (Browser blocked clipboard access)';
                            console.error(err);
                        }}
                    }});
                    </script>
                    """
                    components.html(copy_html, height=50)

                os.remove(out)
            else:
                st.error("Processing failed. Please verify the URL or timestamps.")
