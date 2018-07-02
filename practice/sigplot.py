import wave
import numpy as np
import matplotlib.pyplot as plt
import pyaudio

def main():
    wf = wave.open("2_1.wav", "rb")
    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    CHUNK = wf.getnframes()
    data = wf.readframes(CHUNK)

    while data != None:
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()

    p.terminate()

if __name__ == "__main__":
    main()
