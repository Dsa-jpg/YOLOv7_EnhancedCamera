import cv2


class WebcamView:
    """
    Třída pro innicializaci webkamery a získání snímku z ní.
    """

    def __init__(self, width, height, fps) -> None:
        """
        Konstruktor třídy WebcamView
        :param width: Sířka snímku
        :param height: Výška snímku
        :param fps: Počet snímků za sekundu
        """

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def get_frame(self):
        """
        Metoda pro získání snímku z webkamery
        :return: Snímek z webkamery
        """
        success, img = self.cap.read() # zaznamenání snímku z webkamery
        if not success:
            raise Exception("Failed to capture image")
        return img

    def show_frame(self, img) -> None:
        """
        Metoda pro zobrazení snímku z webkamery
        :param img:  Snímek z webkamery
        :return: Nonw
        """
        cv2.imshow("Webcam", img)

    def release(self) -> None:
        """
        Metoda pro uvolnění webkamery a zavření okna s náhledem
        :return: None
        """

        self.cap.release()
        cv2.destroyAllWindows()
