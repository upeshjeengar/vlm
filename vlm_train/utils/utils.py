import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
def interpolate_color(val, min_val, max_val):
    if max_val - min_val < 1e-6:
        return (255, 255, 255)
    ratio = (val - min_val) / (max_val - min_val)
    r = 255
    g = int(255 * (1 - ratio))
    b = int(255 * (1 - ratio))
    return (r, g, b)

def create_similarity_grid(samples, scores, recall_metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples = len(samples)
    
    # Dimensions
    cell_size = 300 
    header_height = 300 
    header_width = 500
    metrics_height = 300 # Extra space at bottom for text
    
    width = header_width + num_samples * cell_size
    height = header_height + num_samples * cell_size + metrics_height
    
    img_grid = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img_grid)
    
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    if not os.path.exists(font_path):
        font_path = "arial.ttf"
        
    try:
        font_large = ImageFont.truetype(font_path, 60)
        font_medium = ImageFont.truetype(font_path, 30)
        font_small = ImageFont.truetype(font_path, 40)
        font_metrics = ImageFont.truetype(font_path, 40)
    except IOError:
        print("Warning: Could not load requested font. Falling back to default.")
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_metrics = ImageFont.load_default()

    # Draw Column Headers (Images)
    for i in range(num_samples):
        img = samples[i]["orig_image"]
        img.thumbnail((cell_size - 20, header_height - 60))
        x = header_width + i * cell_size + (cell_size - img.size[0]) // 2
        y = (header_height - img.size[1] - 40) // 2
        img_grid.paste(img, (x, y))
        label = f"Image {i}"
        draw.text((x + (img.size[0]//4), y + img.size[1] + 10), label, fill="black", font=font_medium)

    # Draw Rows (Text + Scores)
    for row_idx in range(num_samples):
        text = samples[row_idx]["caption"][:60]
        wrapper_width = 25 
        wrapped_text = ""
        words = text.split()
        line = ""
        for w in words:
            if len(line) + len(w) < wrapper_width:
                line += w + " "
            else:
                wrapped_text += line + "\n"
                line = w + " "
        wrapped_text += line
        
        y_base = header_height + row_idx * cell_size
        draw.text((20, y_base + 40), wrapped_text, fill="black", font=font_small)
        
        row_scores = scores[row_idx, :]
        min_s = row_scores.min().item()
        max_s = row_scores.max().item()
        
        for col_idx in range(num_samples):
            score = scores[row_idx, col_idx].item()
            bg_color = interpolate_color(score, min_s, max_s)
            x_base = header_width + col_idx * cell_size
            shape = [x_base, y_base, x_base + cell_size, y_base + cell_size]
            draw.rectangle(shape, fill=bg_color, outline="black", width=2)
            
            score_txt = f"{score:.3f}"
            left, top, right, bottom = draw.textbbox((0, 0), score_txt, font=font_large)
            text_w = right - left
            text_h = bottom - top
            txt_x = x_base + (cell_size - text_w) // 2
            txt_y = y_base + (cell_size - text_h) // 2
            text_color = "black" if (bg_color[1] > 128) else "white"
            draw.text((txt_x, txt_y), score_txt, fill=text_color, font=font_large)
            
            if row_idx == col_idx:
                draw.rectangle(shape, outline="blue", width=6)

    # --- Draw Metrics at Bottom ---
    metrics_y = header_height + num_samples * cell_size + 40
    
    i2t = recall_metrics["i2t"]
    t2i = recall_metrics["t2i"]
    n_samp = recall_metrics["num_samples"]
    
    lines = [
        f"Evaluation on {n_samp} test samples:",
        f"Image-to-Text: R@1: {i2t[1]:.4f} | R@5: {i2t[5]:.4f} | R@10: {i2t[10]:.4f}",
        f"Text-to-Image: R@1: {t2i[1]:.4f} | R@5: {t2i[5]:.4f} | R@10: {t2i[10]:.4f}"
    ]
    
    for i, line in enumerate(lines):
        draw.text((50, metrics_y + i * 60), line, fill="black", font=font_metrics)

    output_path = os.path.join(output_dir, "similarity_grid.jpg")
    img_grid.save(output_path)
    print(f"Saved grid to {output_path}")

