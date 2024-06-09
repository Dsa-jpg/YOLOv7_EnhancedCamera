from ultralytics import YOLO

class YOLOModel:
    """
    Yolo model načte váhy a umožní predikci na obrázku.
    """
    def __init__(self, weights_path):
        """
        Konstruktor třídy YOLOModel
        :param weights_path:  Cesta k souboru s váhami
        """
        self.model = YOLO(weights_path)

    def predict(self, img):
        """
        Metoda pro predikci na obrázku
        :param img:  Obrázek na kterém se má provést predikce
        :return:  Výsledky predikce na obrázku bounding boxy, třídy a skóre
        """
        return self.model(img)
