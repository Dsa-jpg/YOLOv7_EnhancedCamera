import cv2
from Model import YOLOModel
from Webcam import WebcamView

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]


class YOLOController:
    """
    Třída pro řízení aplikace, která zpracovává snímky z webkamery a zobrazuje je s výsledky predikce.
    """

    def __init__(self, model, view):
        """
        Konstruktor třídy YOLOController
        :param model: Instance třídy YOLOModel
        :param view:  Instance třídy WebcamView
        """

        self.model = model
        self.view = view
        self.zoom_factor = 1.5
        self.smooth_factor = 0.1
        self.target_center = None
        self.prev_target_center = None

    def process_frame(self, img):
        """
        Metoda pro zpracování snímku z webkamery
        :param img:  Snímek z webkamery
        :return:  Snímek z webkamery s výsledky predikce bounding boxů
        """

        results = self.model.predict(img) # predikce na snímku

        people_centers = [] # seznam středů bounding boxů osob

        for r in results:
            boxes = r.boxes # bounding boxy

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0] # souřadnice bounding boxu levy horní roh a pravý dolní roh
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # přetypování na celá čísla

                cls = int(box.cls[0]) # index třídy
                class_name = classNames[cls] # název třídy

                if class_name == "person":
                    person_center = ((x1 + x2) // 2, (y1 + y2) // 2) # střed bounding boxu // je celočíselné dělení
                    people_centers.append(person_center) # přidání středu do seznamu
                    # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), int(0.5))
                    # cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), int(0.5))

        if people_centers:
            avg_center = (
                sum([center[0] for center in people_centers]) // len(people_centers),# všechny x souřadnice a sum / počtem středů
                sum([center[1] for center in people_centers]) // len(people_centers) # všechny y souřadnice a sum / počtem středů
            )


            if self.target_center is None: # pokud je střed cíle None
                self.target_center = avg_center # nastavíme střed cíle na průměrný střed bounding boxů osob
            else:
                # plynulé přesunutí středu cíle
                self.target_center = (
                    int(self.target_center[0] + self.smooth_factor * (avg_center[0] - self.target_center[0])), # nový střed x
                    int(self.target_center[1] + self.smooth_factor * (avg_center[1] - self.target_center[1])) # nový střed y
                )

            if self.prev_target_center:
                cv2.line(img, self.prev_target_center, self.target_center, (0, 0, 0), 1) # vykreslení čáry mezi středy bounding boxů

            self.prev_target_center = self.target_center # uložení středu cíle pro další snímek

            desired_width = self.view.cap.get(cv2.CAP_PROP_FRAME_WIDTH) # šířka snímku
            desired_height = self.view.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # výška snímku

            new_width = int(desired_width // self.zoom_factor) # nová šířka snímku
            new_height = int(desired_height // self.zoom_factor) # nová výška snímku
            x_start = max(self.target_center[0] - new_width // 2, 0) # začátek x souřadnice výřezu
            y_start = max(self.target_center[1] - new_height // 2, 0) # začátek y souřadnice výřezu
            x_end = min(self.target_center[0] + new_width // 2, int(desired_width)) # konec x souřadnice výřezu
            y_end = min(self.target_center[1] + new_height // 2, int(desired_height)) # konec y souřadnice výřezu

            zoomed_img = img[y_start:y_end, x_start:x_end] # výřez snímku kolem středu cíle
            img = cv2.resize(zoomed_img, (int(desired_width), int(desired_height))) # změna velikosti výřezu na původní velikost

        return img

    def run(self):
        """
        Metoda pro spuštění aplikace a zpracování snímků z webkamery
        :return: None
        """
        while True:
            try:
                img = self.view.get_frame() # získání snímku z webkamery
                processed_img = self.process_frame(img) # zpracování snímku
                self.view.show_frame(processed_img) # zobrazení zpracovaného snímku
            except Exception as e:
                print(e)
                break

            if cv2.waitKey(1) == ord('q'): # pokud je stisknuto q ukončíme aplikaci
                break

        self.view.release() # uvolnění webkamery a zavření okna s náhledem


if __name__ == "__main__":
    model = YOLOModel("yolo-Weights/yolov8n.pt") # inicializace modelu
    view = WebcamView(1280, 720, 30) # inicializace webkamery
    controller = YOLOController(model, view) # inicializace řídicí třídy
    controller.run() # spuštění aplikace
