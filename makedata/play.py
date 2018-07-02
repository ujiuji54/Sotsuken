import wave
import pyaudio

def play():
    wf = wave.open("somewave.wav", "rb")
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
    wf.close()

    p.terminate()
