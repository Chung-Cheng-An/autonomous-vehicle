import time
import cv2
import numpy as np
import speech_recognition as sr
from gpiozero import Motor, DistanceSensor, PWMOutputDevice, LED
import tflite_runtime.interpreter as tflite
from threading import Thread
from luma.core.interface.serial import spi, noop
from luma.led_matrix.device import max7219
from luma.core.render import canvas

# ==========================
# 模式指示 LED（新增）
# ==========================
led_mode1 = LED(5)   # 模式1：語音
led_mode2 = LED(6)   # 模式2：手勢

def set_mode_led(mode):
    if mode == "voice":
        led_mode1.on()
        led_mode2.off()
    elif mode == "gesture":
        led_mode1.off()
        led_mode2.on()
    else:
        led_mode1.off()
        led_mode2.off()


# ==========================
# MAX7219 初始化
# ==========================
serial = spi(port=0, device=0, gpio=noop())
matrix = max7219(serial, cascaded=1)
matrix.command(0x09, 0x00)
matrix.command(0x0A, 0x08)
matrix.command(0x0B, 0x07)
matrix.command(0x0C, 0x01)
matrix.command(0x0F, 0x00)
matrix.clear()
time.sleep(0.05)

# ==========================
# 箭頭點陣
# ==========================
ARROW_UP = [
    0b00011000,
    0b00111100,
    0b01111110,
    0b00011000,
    0b00011000,
    0b00011000,
    0b00011000,
    0b00000000
]

ARROW_DOWN = [
    0b00011000,
    0b00011000,
    0b00011000,
    0b00011000,
    0b01111110,
    0b00111100,
    0b00011000,
    0b00000000
]

ARROW_LEFT = [
    0b00010000,
    0b00110000,
    0b01111111,
    0b01111111,
    0b00110000,
    0b00010000,
    0b00000000,
    0b00000000
]

ARROW_RIGHT = [
    0b00001000,
    0b00001100,
    0b11111110,
    0b11111110,
    0b00001100,
    0b00001000,
    0b00000000,
    0b00000000
]

ARROW_STOP = [
    0b01111110,
    0b01000010,
    0b01000010,
    0b01000010,
    0b01000010,
    0b01000010,
    0b01111110,
    0b00000000
]

def show_arrow(pattern):
    matrix.clear()
    with canvas(matrix) as draw:
        for y, row in enumerate(pattern):
            for x in range(8):
                if row & (1 << (7 - x)):
                    draw.point((x, y), fill="white")

# ==========================
# 馬達 + 超音波 + 蜂鳴器
# ==========================
CAR_STATE = "stop"
REAR_BLOCK = False

right_motor = Motor(forward=23, backward=24)
left_motor  = Motor(forward=17, backward=27)

ultrasonic = DistanceSensor(trigger=14, echo=15, max_distance=2.0, queue_len=3)
buzzer = PWMOutputDevice(12)
buzzer.value = 0

def safe_action(action, delay=0.12):
    right_motor.stop()
    left_motor.stop()
    time.sleep(delay)
    action()

# ==========================
# 車子動作
# ==========================
def stop():
    global CAR_STATE
    CAR_STATE = "stop"
    show_arrow(ARROW_STOP)
    right_motor.stop()
    left_motor.stop()

def forward():
    global CAR_STATE
    CAR_STATE = "forward"
    show_arrow(ARROW_UP)
    safe_action(lambda: (right_motor.forward(), left_motor.forward()))

def backward():
    global CAR_STATE
    if REAR_BLOCK:
        print("Rear obstacle too close, braking")
        stop()
        return
    CAR_STATE = "backward"
    show_arrow(ARROW_DOWN)
    safe_action(lambda: (right_motor.backward(), left_motor.backward()))

def turnLeft():
    global CAR_STATE
    CAR_STATE = "left"
    show_arrow(ARROW_LEFT)
    safe_action(lambda: (left_motor.stop(), right_motor.forward()))

def turnRight():
    global CAR_STATE
    CAR_STATE = "right"
    show_arrow(ARROW_RIGHT)
    safe_action(lambda: (left_motor.forward(), right_motor.stop()))

# ==========================
# 距離監控
# ==========================
def distance_monitor():
    global REAR_BLOCK

    while True:
        distance_cm = ultrasonic.distance * 100
        print(f"Distance: {distance_cm:.1f} cm")  # 一直顯示距離

        if CAR_STATE == "backward":
            if distance_cm < 15:
                REAR_BLOCK = True
                stop()
                for _ in range(3):
                    buzzer.value = 0.7
                    time.sleep(0.1)
                    buzzer.value = 0
                    time.sleep(0.1)
            elif distance_cm < 30:
                REAR_BLOCK = False
                buzzer.value = 0.5
                time.sleep(0.2)
                buzzer.value = 0
                time.sleep(0.3)
            else:
                REAR_BLOCK = False
                buzzer.value = 0
                time.sleep(0.1)
        else:
            REAR_BLOCK = False
            buzzer.value = 0
            time.sleep(0.05)

Thread(target=distance_monitor, daemon=True).start()

# ==========================
# 手勢控制
# ==========================
GESTURE_MAP = {0:"forward",1:"backward",2:"left",3:"right",4:"stop"}
CONF_THRESHOLD = 0.9
REQUIRED_CONFIRM = 3

class GestureController:
    def __init__(self):
        self.model_path = "gesture.tflite"
        self.last_cmd = None
        self.confirm_cmd = None
        self.confirm_count = 0
        self.cap = None

        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.h = self.input_details[0]["shape"][1]
        self.w = self.input_details[0]["shape"][2]

    def _preprocess(self, frame):
        frame = cv2.resize(frame, (self.w, self.h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32)/255.0
        return np.expand_dims(frame, axis=0)

    def _execute(self, cmd):
        if cmd=="forward": forward()
        elif cmd=="backward": backward()
        elif cmd=="left": turnLeft()
        elif cmd=="right": turnRight()
        else: stop()

    def run(self):
        stop()
        self.cap = cv2.VideoCapture(0)
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret: continue
                input_data = self._preprocess(frame)
                self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
                self.interpreter.invoke()
                output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

                idx = int(np.argmax(output))
                conf = float(output[idx])
                cmd = GESTURE_MAP.get(idx, "停止")

                if conf >= CONF_THRESHOLD:
                    if cmd==self.confirm_cmd: self.confirm_count+=1
                    else: self.confirm_cmd = cmd; self.confirm_count=1
                else:
                    self.confirm_count=0

                if self.confirm_count>=REQUIRED_CONFIRM and cmd!=self.last_cmd:
                    self._execute(cmd)
                    self.last_cmd=cmd

                cv2.putText(frame,f"{cmd} | conf={conf:.2f} | cnt={self.confirm_count}",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
                cv2.imshow("Gesture Mode", frame)
                if cv2.waitKey(1)==27: break
        finally:
            stop()
            if self.cap: self.cap.release()
            cv2.destroyAllWindows()

# ==========================
# 語音控制
# ==========================
recognizer = sr.Recognizer()
microphone = sr.Microphone()

def process_voice(cmd):
    if "模式2" in cmd: 
                set_mode_led("gesture")
                return "gesture"
    elif "前進" in cmd: forward()
    elif "後退" in cmd: backward()
    elif "左轉" in cmd: turnLeft()
    elif "右轉" in cmd: turnRight()
    elif "停止" in cmd: stop()
    return "voice"

# ==========================
# 主程式
# ==========================
def main():
    mode = "voice"
    set_mode_led("voice") #模式一  亮燈
    gesture_ctrl = GestureController()

    with microphone as source: recognizer.adjust_for_ambient_noise(source, duration=1)
    print("System started | Say '模式2' for gesture mode | ESC to exit")

    while True:
        try:
            if mode=="voice":
                with microphone as source:
                    audio = recognizer.listen(source, phrase_time_limit=3)
                cmd = recognizer.recognize_google(audio, language="zh-TW")
                mode = process_voice(cmd)
            elif mode=="gesture":
                gesture_ctrl.run()
                set_mode_led("voice")
                mode="voice"
        except KeyboardInterrupt:
            stop()
            buzzer.off()
            set_mode_led("None")
            break
        except Exception as e:
            print("Error:", e)

if __name__=="__main__":
    main()
