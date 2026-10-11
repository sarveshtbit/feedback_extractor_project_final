import os
import json
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Reference Canvas Dimensions at 300 DPI
CANVAS_WIDTH = 2479
CANVAS_HEIGHT = 3508
DPI = 300

# Color definitions (RGB)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY_TEXT = (80, 80, 80)
COLOR_BOX_BORDER = (100, 100, 100)
COLOR_GUIDE_DOTS = (180, 200, 240)  # Light blue/gray for color-threshold removal testing

def create_form_template(output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), COLOR_WHITE)
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 64)
        font_header = ImageFont.truetype("arial.ttf", 48)
        font_label = ImageFont.truetype("arial.ttf", 36)
        font_sublabel = ImageFont.truetype("arial.ttf", 28)
    except IOError:
        font_title = font_header = font_label = font_sublabel = ImageFont.load_default()

    config = {
        "canvas": {
            "width": CANVAS_WIDTH,
            "height": CANVAS_HEIGHT,
            "dpi": DPI
        },
        "registration_markers": {},
        "fields": {}
    }

    # 1. Four Corner Registration Markers (80x80 px squares with center crosshairs)
    marker_size = 80
    markers = {
        "top_left": (100, 100),
        "top_right": (CANVAS_WIDTH - 100 - marker_size, 100),
        "bottom_left": (100, CANVAS_HEIGHT - 100 - marker_size),
        "bottom_right": (CANVAS_WIDTH - 100 - marker_size, CANVAS_HEIGHT - 100 - marker_size)
    }

    for key, (mx, my) in markers.items():
        # Outer solid square
        draw.rectangle([mx, my, mx + marker_size, my + marker_size], fill=COLOR_BLACK)
        # Inner white square for high-contrast alignment
        margin = 16
        draw.rectangle([mx + margin, my + margin, mx + marker_size - margin, my + marker_size - margin], fill=COLOR_WHITE)
        # Center black dot
        dot_margin = 32
        draw.rectangle([mx + dot_margin, my + dot_margin, mx + marker_size - dot_margin, my + marker_size - dot_margin], fill=COLOR_BLACK)
        
        cx = mx + marker_size // 2
        cy = my + marker_size // 2
        config["registration_markers"][key] = {
            "x": cx,
            "y": cy,
            "box": [mx, my, marker_size, marker_size]
        }

    # 2. Header
    title_text = "BANNARI AMMAN INSTITUTE OF TECHNOLOGY"
    draw.text((CANVAS_WIDTH // 2 - 650, 220), title_text, fill=COLOR_BLACK, font=font_title)
    
    subtitle_text = "STUDENT ADMISSION & FEEDBACK FORM"
    draw.text((CANVAS_WIDTH // 2 - 480, 310), subtitle_text, fill=COLOR_GRAY_TEXT, font=font_header)

    draw.line([200, 390, CANVAS_WIDTH - 200, 390], fill=COLOR_BLACK, width=4)

    current_y = 450

    # Helper function for drawing boxes with guide dots
    def draw_character_boxes(label, field_id, count, start_y, box_w=110, box_h=140, gap=15, is_cutoff=False):
        draw.text((200, start_y), label, fill=COLOR_BLACK, font=font_label)
        box_start_y = start_y + 50
        box_start_x = 200
        
        field_boxes = []
        for i in range(count):
            bx = box_start_x + i * (box_w + gap)
            by = box_start_y
            
            if is_cutoff and i == 3:  # Decimal point marker
                draw.ellipse([bx + 15, by + box_h - 35, bx + 35, by + box_h - 15], fill=COLOR_BLACK)
                field_boxes.append({
                    "index": i,
                    "type": "decimal",
                    "box": [bx, by, box_w // 2, box_h]
                })
                box_start_x -= (box_w // 2)  # Adjust subsequent offset
                continue

            # Box outer border
            draw.rectangle([bx, by, bx + box_w, by + box_h], outline=COLOR_BOX_BORDER, width=3)

            # Draw light blue/gray guide dots inside the box (4 dots)
            dot_r = 3
            dots = [
                (bx + box_w // 4, by + box_h // 4),
                (bx + 3 * box_w // 4, by + box_h // 4),
                (bx + box_w // 4, by + 3 * box_h // 4),
                (bx + 3 * box_w // 4, by + 3 * box_h // 4),
            ]
            for dx, dy in dots:
                draw.ellipse([dx - dot_r, dy - dot_r, dx + dot_r, dy + dot_r], fill=COLOR_GUIDE_DOTS)

            field_boxes.append({
                "index": i,
                "type": "box",
                "box": [bx, by, box_w, box_h]
            })

        total_w = field_boxes[-1]["box"][0] + field_boxes[-1]["box"][2] - 200
        config["fields"][field_id] = {
            "type": "character_boxes",
            "count": count,
            "bounding_box": [200, box_start_y, total_w, box_h],
            "boxes": field_boxes
        }

        return box_start_y + box_h + 60

    # Field 1: Mobile Number (Priority #1)
    current_y = draw_character_boxes("1. STUDENT MOBILE NUMBER (10 Digits - Mandatory):", "mobile_number", 10, current_y, box_w=110, box_h=140, gap=15)

    # Field 2: Student Name
    current_y = draw_character_boxes("2. STUDENT FULL NAME (In Block Letters):", "student_name", 15, current_y, box_w=110, box_h=140, gap=15)

    # Field 3: Cutoff Mark (e.g. 198.50)
    current_y = draw_character_boxes("3. HIGHER SECONDARY CUTOFF MARK (e.g. 198.50):", "cutoff_mark", 6, current_y, box_w=100, box_h=130, gap=12, is_cutoff=True)

    # Field 4: School Name
    current_y = draw_character_boxes("4. SCHOOL NAME / CITY:", "school_name", 18, current_y, box_w=95, box_h=120, gap=12)

    # Field 5: OMR Feedback Section
    draw.text((200, current_y), "5. INSTITUTION FEEDBACK (Fill bubbles completely):", fill=COLOR_BLACK, font=font_label)
    omr_start_y = current_y + 60
    
    questions = [
        ("q1", "Teaching Quality & Faculty Support"),
        ("q2", "Laboratories & Infrastructure"),
        ("q3", "Library & Digital Resources"),
        ("q4", "Hostel & Dining Facilities"),
        ("q5", "Overall Campus Experience")
    ]
    options_labels = ["A (Ex)", "B (Good)", "C (Avg)", "D (Poor)"]

    omr_config = {"type": "omr_group", "questions": []}
    qy = omr_start_y

    for q_id, q_text in questions:
        draw.text((220, qy), f"{q_id.upper()}. {q_text}", fill=COLOR_BLACK, font=font_sublabel)
        q_options = []
        
        opt_start_x = 1100
        for opt_idx, opt_text in enumerate(options_labels):
            bx = opt_start_x + opt_idx * 300
            by = qy + 15
            radius = 28

            # Draw outer bubble circle
            draw.ellipse([bx - radius, by - radius, bx + radius, by + radius], outline=COLOR_BLACK, width=4)
            # Label
            draw.text((bx - 12, by - 14), opt_text[0], fill=COLOR_BLACK, font=font_sublabel)

            q_options.append({
                "label": opt_text[0],
                "center": [bx, by],
                "radius": radius
            })

        omr_config["questions"].append({
            "question_id": q_id,
            "label": q_text,
            "options": q_options
        })
        qy += 90

    config["fields"]["omr_feedback"] = omr_config

    # Save PNG template
    template_png_path = os.path.join(output_dir, "bannari_amman_form_template.png")
    img.save(template_png_path, "PNG", dpi=(DPI, DPI))
    print(f"Generated template PNG: {template_png_path}")

    # Save Config JSON (Single source of truth)
    config_json_path = os.path.join(output_dir, "form_config.json")
    with open(config_json_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Generated form config JSON: {config_json_path}")

    # Save PDF version using ReportLab
    pdf_path = os.path.join(output_dir, "bannari_amman_form_template.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    # Draw high-res image onto A4 PDF page
    a4_w, a4_h = A4
    c.drawImage(template_png_path, 0, 0, width=a4_w, height=a4_h)
    c.showPage()
    c.save()
    print(f"Generated template PDF: {pdf_path}")

    return config_json_path, template_png_path, pdf_path

if __name__ == "__main__":
    create_form_template(".")
