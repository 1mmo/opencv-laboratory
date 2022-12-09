import cv2 as cv
from managers import WindowManager, CaptureManager
from filters import ReductionDetail, Quant
from Fourier import sobel

class PixelArt():
    def __init__(self):
        self._windowManager = WindowManager('PixelArt', self.onKeypress)
        self._captureManger = CaptureManager(cv.VideoCapture(0),
                                             self._windowManager,
                                             True)
        self.filter = None
        self.use_filter = None

    def run(self):
        """ Run the main loop. """
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManger.enterFrame()
            frame = self._captureManger.frame
            if frame is not None and self.use_filter is not None:
                self._captureManger._frame = self.filter.reduction(image=frame)
            self._captureManger.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """
        Handle a keypress.
        space -> Take a screenshot.
        tap -> Start/stop pixeling.
        escape -> Quit.
        """
        if keycode == 32: # space
            self._captureManger.writeImage('screenshot.png')
        elif keycode == 49: # keycode number 1
            if self.use_filter is None:
                self.filter = ReductionDetail()
                self.use_filter = True
            else:
                self.use_filter = None
        elif keycode == 50: # keycode number 2
            if self.use_filter is None:
                self.filter = Quant()
                self.use_filter = True
            else:
                self.use_filter = None
        elif keycode == 27: # escape
            self._windowManager.destroyWindow()

if __name__ == "__main__":
    PixelArt().run()

