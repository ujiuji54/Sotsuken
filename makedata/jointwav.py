import wave

def join_waves(inputs, output):
    '''
    inputs : list of filenames
    output : output filename
    '''
    fps = [wave.open(f, 'r') for f in inputs]
    fpw = wave.open(output, 'w')

    fpw.setnchannels(fps[0].getnchannels())
    fpw.setsampwidth(fps[0].getsampwidth())
    fpw.setframerate(fps[0].getframerate())

    for fp in fps:
        fpw.writeframes(fp.readframes(fp.getnframes()))
        fp.close()
    fpw.close()

if __name__ == '__main__':
    inputs = ['train_x' + str(n) + '.wav' for n in range(9)]
    output = 'joint.wav'

    join_waves(inputs, output)
