import cv2
import numpy as np
import logging
import sys
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import requests  # Импорт для работы с HTTP запросами

# ГЛОБАЛЫЕ НАСТРОЙКИ
MODEL_PATH = "yolov8n.pt"      # Путь к модели YOLO
VIDEO_SOURCE = "4.mp4"         # Источник видео
IOU_POROG = 0.5                # Порог IoU для сопоставления объектов
SWITCH_SVETOFOR = 2.0          # Время (сек) до переключения светофора с RED на GREEN
GREEN_DURATION = 2.0          # Длительность зеленого сигнала (сек)
UPDATE_DELAY = 10              # Задержка обновления кадра (мс)
ARDUINO_BASE_URL = "http://192.168.0.119"  # IP Arduino

# Начальные размеры приложения
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

logging.getLogger("ultralytics").setLevel(logging.ERROR)

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

def find_matching_object(detection_box, tracked_objects, IOU_POROG=IOU_POROG):
    for obj_id, data in tracked_objects.items():
        if compute_iou(data["coords"], detection_box) > IOU_POROG:
            return obj_id
    return None

def send_arduino_command(command):

    url = ARDUINO_BASE_URL + command
    try:
        response = requests.get(url, timeout=1)
        if response.status_code == 200:
            print(f"Команда {command} успешно отправлена.")
        else:
            print(f"Ошибка при отправке команды {command}: статус {response.status_code}")
    except Exception as e:
        print(f"Ошибка при обращении к Arduino: {e}")

class ZoneSelector:
    def __init__(self, canvas, canvas_width, canvas_height):
        self.canvas = canvas
        self.points = []
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.polygon_id = None

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<BackSpace>", self.delete_last_point)

    def on_click(self, event):
        x, y = event.x, event.y
        if self.points and len(self.points) >= 3:
            first_point = self.points[0]
            threshold = 10
            if abs(x - first_point[0]) < threshold and abs(y - first_point[1]) < threshold:
                self.on_right_click(event)
                self.canvas.unbind("<Button-1>")
                self.canvas.unbind("<Button-3>")
                self.canvas.unbind("<BackSpace>")
                return
        self.points.append((x, y))
        r = 3
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="blue", tags="points")
        if len(self.points) > 1:
            self.canvas.create_line(self.points[-2][0], self.points[-2][1], x, y, fill="blue", width=2, tags="points")

    def on_right_click(self, event):
        if len(self.points) >= 3:
            self.canvas.create_line(self.points[-1][0], self.points[-1][1],
                                     self.points[0][0], self.points[0][1], fill="blue", width=2, tags="points")
            if self.polygon_id:
                self.canvas.delete(self.polygon_id)
            self.polygon_id = self.canvas.create_polygon(self.points, outline="blue", fill="", width=2, tags="points")
        else:
            print("Для выделения зоны необходимо минимум 3 точки.")

    def delete_last_point(self, event):
        if self.points:
            self.points.pop()
            self.canvas.delete("points")
            if self.points:
                for idx, (x, y) in enumerate(self.points):
                    r = 3
                    self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="blue", tags="points")
                    if idx > 0:
                        self.canvas.create_line(self.points[idx - 1][0], self.points[idx - 1][1], x, y, fill="blue",
                                                 width=2, tags="points")
        else:
            print("Больше точек для удаления нет.")

    def get_mask(self):
        mask = np.zeros((self.canvas_height, self.canvas_width), dtype=np.uint8)
        if len(self.points) >= 3:
            pts = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        return mask

class TrafficLightApp:
    def __init__(self, root, video_label, light_canvas, btn_select_zone):
        self.root = root
        self.video_label = video_label
        self.light_canvas = light_canvas
        self.btn_select_zone = btn_select_zone

        self.model = YOLO(MODEL_PATH)
        self.cap = cv2.VideoCapture(VIDEO_SOURCE)
        ret, frame = self.cap.read()
        if not ret:
            print("Ошибка открытия видео!")
            sys.exit(1)

        self.frame_height, self.frame_width = frame.shape[:2]
        self.default_display_width = DEFAULT_WIDTH
        self.default_display_height = DEFAULT_HEIGHT

        self.mask = None

        display_frame = cv2.resize(frame, (self.default_display_width, self.default_display_height))
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb_image))
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk

        self.tracked_objects = {}
        self.next_obj_id = 0

        current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self.crossing_start_time = None
        self.red_start_time = current_time
        self.green_start_time = current_time
        self.traffic_light_state = "RED"

        # Начальное состояние Arduino: красный включён, зеленый выключен
        send_arduino_command("/14/off")
        send_arduino_command("/12/on")

        self.zone_selector = None
        self.updating = False

        self.btn_select_zone.config(command=self.select_zone)

    def update(self):
        if not self.updating:
            self.root.after(UPDATE_DELAY, self.update)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            return

        current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        dynamic_width = self.video_label.winfo_width()
        dynamic_height = self.video_label.winfo_height()
        if dynamic_width < 50 or dynamic_height < 50:
            dynamic_width = self.default_display_width
            dynamic_height = self.default_display_height

        timer_x = int(self.frame_width * 0.02)
        timer_y = int(self.frame_height * 0.1)
        font_scale = (self.frame_height / 480.0) * 0.75
        thickness = max(1, int(font_scale * 2))

        mask_resized = cv2.resize(self.mask, (frame.shape[1], frame.shape[0]))
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask_resized)
        mask_color = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        mask_color[mask_resized > 0] = [0, 0, 255]
        overlay = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)

        results = self.model(masked_frame, verbose=False)
        result = results[0]
        for box in result.boxes:
            if int(box.cls.item()) == 0:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if mask_resized[center[1], center[0]] > 0:
                    obj_id = find_matching_object((x1, y1, x2, y2), self.tracked_objects)
                    if obj_id is None:
                        if self.crossing_start_time is None:
                            self.crossing_start_time = current_time
                        obj_id = self.next_obj_id
                        label = chr(65 + (self.next_obj_id % 26))
                        self.tracked_objects[obj_id] = {
                            "first_seen_time": self.crossing_start_time,
                            "last_seen_time": current_time,
                            "positions": [center],
                            "label": label,
                            "coords": (x1, y1, x2, y2)
                        }
                        self.next_obj_id += 1
                    else:
                        self.tracked_objects[obj_id]["last_seen_time"] = current_time
                        self.tracked_objects[obj_id]["positions"].append(center)
                        if len(self.tracked_objects[obj_id]["positions"]) > 20:
                            self.tracked_objects[obj_id]["positions"].pop(0)
                        self.tracked_objects[obj_id]["coords"] = (x1, y1, x2, y2)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(overlay, self.tracked_objects[obj_id]["label"], (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                    positions = self.tracked_objects[obj_id]["positions"]
                    if len(positions) > 1:
                        pts = np.array(positions, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(overlay, [pts], False, (255, 0, 0), thickness)

        for obj_id in list(self.tracked_objects.keys()):
            if current_time - self.tracked_objects[obj_id]["last_seen_time"] > 1.0:
                del self.tracked_objects[obj_id]
                if not self.tracked_objects:
                    self.crossing_start_time = None

        # Если объектов нет, сбрасываем таймер, чтобы red_elapsed не накапливался
        if not self.tracked_objects:
            self.red_start_time = current_time

        if self.tracked_objects:
            if self.traffic_light_state == "RED" and (current_time - self.red_start_time) >= SWITCH_SVETOFOR:
                self.traffic_light_state = "GREEN"
                self.green_start_time = current_time
                send_arduino_command("/14/on")
                send_arduino_command("/12/off")
            elif self.traffic_light_state == "GREEN" and (current_time - self.green_start_time) >= GREEN_DURATION:
                self.traffic_light_state = "RED"
                self.red_start_time = current_time
                send_arduino_command("/12/on")
                send_arduino_command("/14/off")
        else:
            # Если объектов нет, всегда обновляем красный таймер и переводим в RED, если необходимо
            self.red_start_time = current_time
            if self.traffic_light_state != "RED":
                self.traffic_light_state = "RED"
                send_arduino_command("/14/off")
                send_arduino_command("/12/on")

        if self.traffic_light_state == "RED":
            red_elapsed = current_time - self.red_start_time
            green_elapsed = 0.0
        elif self.traffic_light_state == "GREEN":
            green_elapsed = current_time - self.green_start_time
            red_elapsed = 0.0
        else:
            red_elapsed = 0.0
            green_elapsed = 0.0

        cv2.putText(overlay, f"Red: {red_elapsed:.1f}s", (timer_x, timer_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
        cv2.putText(overlay, f"Green: {green_elapsed:.1f}s", (timer_x, timer_y + int(30 * font_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        display_frame = cv2.resize(overlay, (dynamic_width, dynamic_height))
        rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb_display))
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk

        self.update_traffic_light_canvas()
        self.root.after(UPDATE_DELAY, self.update)

    def update_traffic_light_canvas(self):
        self.light_canvas.delete("all")
        if self.traffic_light_state == "RED":
            top_color, bottom_color = "red", "grey"
        elif self.traffic_light_state == "GREEN":
            top_color, bottom_color = "grey", "green"
        else:
            top_color, bottom_color = "grey", "grey"
        self.light_canvas.create_oval(50, 50, 150, 150, fill=top_color, outline="black", width=2)
        self.light_canvas.create_oval(50, 170, 150, 270, fill=bottom_color, outline="black", width=2)

    def select_zone(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Невозможно получить кадр для выделения зоны.")
            return

        current_width = self.video_label.winfo_width()
        current_height = self.video_label.winfo_height()
        if current_width < 50 or current_height < 50:
            current_width = self.default_display_width
            current_height = self.default_display_height

        zone_window = tk.Toplevel(self.root)
        zone_window.title("Выделите зону")
        canvas = tk.Canvas(zone_window, width=current_width, height=current_height, bg="white")
        canvas.pack()

        display_frame = cv2.resize(frame, (current_width, current_height))
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=pil_image)
        canvas.create_image(0, 0, anchor="nw", image=imgtk, tags="bg")
        canvas.imgtk = imgtk

        self.zone_selector = ZoneSelector(canvas, current_width, current_height)

        def finish_zone():
            zone_mask = self.zone_selector.get_mask()
            if zone_mask is not None and np.count_nonzero(zone_mask) > 0:
                new_mask = cv2.resize(zone_mask, (self.frame_width, self.frame_height), interpolation=cv2.INTER_NEAREST)
                self.mask = new_mask
                print("Зона выделена и маска обновлена.")
                self.updating = True
                self.update()
            else:
                print("Зона не выделена. Видео останется на паузе.")
            zone_window.destroy()

        btn_finish = tk.Button(zone_window, text="Завершить выбор", command=finish_zone)
        btn_finish.pack()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Traffic Light Application")

    app_frame = tk.Frame(root)
    app_frame.pack(fill="both", expand=True)

    video_label = tk.Label(app_frame, bg="black")
    video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    right_frame = tk.Frame(app_frame)
    right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")
    light_canvas = tk.Canvas(right_frame, width=200, height=350, bg="white")
    light_canvas.pack(pady=10)
    btn_select_zone = tk.Button(right_frame, text="Выбрать зону")
    btn_select_zone.pack(pady=10)

    app_frame.columnconfigure(0, weight=1)
    app_frame.rowconfigure(0, weight=1)

    app = TrafficLightApp(root, video_label, light_canvas, btn_select_zone)
    root.mainloop()
