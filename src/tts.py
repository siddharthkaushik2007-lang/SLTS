import queue
import threading
import pyttsx3


class TextToSpeech:
    def __init__(self, rate: int = 150, volume: float = 1.0):
        self.rate = rate
        self.volume = volume
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        engine.startLoop(False)

        try:
            while True:
                try:
                    text = self._queue.get(timeout=0.01)
                except queue.Empty:
                    engine.iterate()
                    if self._stop_event.is_set():
                        break
                    continue

                if text is None:
                    self._queue.task_done()
                    break

                engine.say(text)
                self._queue.task_done()
                engine.iterate()
        finally:
            engine.endLoop()
            engine.stop()

    def speak(self, text: str):
        """Queue text to be spoken without blocking the webcam loop."""
        cleaned = text.strip()
        if cleaned:
            self._queue.put(cleaned)

    def close(self):
        """Cleanly shut down the TTS worker thread."""
        self._stop_event.set()
        self._queue.put(None)
        self._thread.join(timeout=2)


if __name__ == "__main__":
    import time

    tts = TextToSpeech()
    tts.speak("Hello, sign language text to speech is working.")
    time.sleep(4)
    tts.close()